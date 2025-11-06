import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import random

# 设备检测和设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

if torch.cuda.is_available():
    print("🎯 使用GPU进行训练")
    # GPU优化设置
    torch.backends.cudnn.benchmark = True  # 加速卷积层
    torch.backends.cudnn.deterministic = False  # 为了速度牺牲可重复性
else:
    print("⚡ 使用CPU进行训练")

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PolicyDataset(Dataset):
    def __init__(self, texts, intents=None, entities=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.intents = intents
        self.entities = entities
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 意图标签映射
        self.intent_labels = [
            'query_subsidy', 'application_process', 'product_scope',
            'qualification_check', 'document_requirements', 'deadline_query',
            'regional_policy', 'appeal_process', 'policy_comparison', 'other'
        ]
        self.intent2id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id2intent = {idx: label for label, idx in self.intent2id.items()}
        
        # 实体标签映射
        self.entity_labels = ['O', 'B-SUBSIDY', 'I-SUBSIDY', 'B-PRODUCT', 'I-PRODUCT',
                             'B-LOCATION', 'I-LOCATION', 'B-TIME', 'I-TIME',
                             'B-CONDITION', 'I-CONDITION', 'B-DOCUMENT', 'I-DOCUMENT']
        self.entity2id = {label: idx for idx, label in enumerate(self.entity_labels)}
        self.id2entity = {idx: label for label, idx in self.entity2id.items()}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        # 添加意图标签
        if self.intents is not None:
            intent = self.intents[idx]
            intent_id = self.intent2id.get(intent, self.intent2id['other'])
            output['intent_labels'] = torch.tensor(intent_id, dtype=torch.long)
        
        # 添加实体标签
        if self.entities is not None:
            entity_tags = self.entities[idx]
            # 将实体标签转换为ID
            entity_ids = []
            for i, tag in enumerate(entity_tags):
                if i < self.max_length:
                    entity_ids.append(self.entity2id.get(tag, self.entity2id['O']))
            
            # 填充到最大长度
            while len(entity_ids) < self.max_length:
                entity_ids.append(self.entity2id['O'])
                
            output['entity_labels'] = torch.tensor(entity_ids[:self.max_length], dtype=torch.long)
        
        return output

class PolicyMultiTaskModel(nn.Module):
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', num_intents=10, num_entities=13):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size
        
        # 意图分类器
        self.intent_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_intents)
        )
        
        # 实体识别器
        self.entity_recognizer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_entities)
        )
        
        # 损失函数
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.entity_loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        
    def forward(self, input_ids, attention_mask, intent_labels=None, entity_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # 意图分类
        intent_logits = self.intent_classifier(pooled_output)
        
        # 实体识别
        entity_logits = self.entity_recognizer(sequence_output)
        
        loss = 0
        if intent_labels is not None and entity_labels is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)
            entity_loss = self.entity_loss_fn(
                entity_logits.view(-1, entity_logits.size(-1)), 
                entity_labels.view(-1)
            )
            loss = intent_loss + 0.8 * entity_loss
        
        return {
            'loss': loss,
            'intent_logits': intent_logits,
            'entity_logits': entity_logits
        }

class PolicyTrainer:
    def __init__(self, model, train_loader, val_loader, learning_rate=2e-5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)
        
        # 如果有多GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        intent_correct = 0
        intent_total = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移动到设备（GPU或CPU）
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            intent_labels = batch['intent_labels'].to(self.device, non_blocking=True)
            entity_labels = batch['entity_labels'].to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                intent_labels=intent_labels,
                entity_labels=entity_labels
            )
            
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 计算意图准确率
            intent_preds = torch.argmax(outputs['intent_logits'], dim=1)
            intent_correct += (intent_preds == intent_labels).sum().item()
            intent_total += intent_labels.size(0)
            
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024**3
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}, GPU Mem: {gpu_memory:.2f}GB')
                else:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}')
        
        avg_loss = total_loss / len(self.train_loader)
        intent_acc = intent_correct / intent_total
        return avg_loss, intent_acc
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        intent_correct = 0
        intent_total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                intent_labels = batch['intent_labels'].to(self.device, non_blocking=True)
                entity_labels = batch['entity_labels'].to(self.device, non_blocking=True)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    intent_labels=intent_labels,
                    entity_labels=entity_labels
                )
                
                total_loss += outputs['loss'].item()
                intent_preds = torch.argmax(outputs['intent_logits'], dim=1)
                intent_correct += (intent_preds == intent_labels).sum().item()
                intent_total += intent_labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        intent_acc = intent_correct / intent_total
        return avg_loss, intent_acc
    
    def train(self, epochs=3):
        print("开始训练政策咨询智能体...")
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            self.scheduler.step()
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
            print(f'  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # 保存模型时移除DataParallel包装
                model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_val_loss,
                    'device': str(self.device)
                }, 'best_policy_model.pth')
                print(f'  保存最佳模型到 best_policy_model.pth')
            
            print('-' * 60)

# 数据生成函数
def generate_policy_data():
    """生成模拟的政策咨询训练数据"""
    
    # 示例问题和对应的意图
    intent_data = [
        # 补贴查询
        ("汽车以旧换新补贴多少钱？", "query_subsidy"),
        ("家电补贴标准是多少？", "query_subsidy"),
        ("手机以旧换新能补贴多少？", "query_subsidy"),
        ("新能源汽车补贴政策", "query_subsidy"),
        ("以旧换新补贴金额", "query_subsidy"),
        
        # 申请流程
        ("怎么申请家电补贴？", "application_process"),
        ("汽车以旧换新申请步骤", "application_process"),
        ("补贴申请需要哪些步骤？", "application_process"),
        ("线上申请流程是怎样的？", "application_process"),
        ("申请补贴的具体流程", "application_process"),
        
        # 产品范围
        ("哪些手机可以参与以旧换新？", "product_scope"),
        ("支持以旧换新的家电类型", "product_scope"),
        ("哪些汽车品牌参与活动？", "product_scope"),
        ("数码产品包括哪些？", "product_scope"),
        ("参与以旧换新的产品范围", "product_scope"),
        
        # 资格检查
        ("我符合补贴条件吗？", "qualification_check"),
        ("申请需要什么资格？", "qualification_check"),
        ("外地户口可以申请吗？", "qualification_check"),
        ("企业可以参与吗？", "qualification_check"),
        ("个人申请条件是什么？", "qualification_check"),
        
        # 材料要求
        ("我需要准备什么材料？", "document_requirements"),
        ("申请需要哪些证件？", "document_requirements"),
        ("要提交什么证明文件？", "document_requirements"),
        ("材料清单有哪些？", "document_requirements"),
        ("需要准备哪些申请材料？", "document_requirements"),
        
        # 截止时间
        ("申请截止到什么时候？", "deadline_query"),
        ("活动持续到哪天？", "deadline_query"),
        ("补贴政策有效期", "deadline_query"),
        ("什么时候截止申请？", "deadline_query"),
        ("政策执行到何时？", "deadline_query"),
        
        # 地区政策
        ("北京地区的补贴政策", "regional_policy"),
        ("上海以旧换新标准", "regional_policy"),
        ("广州有什么特殊政策？", "regional_policy"),
        ("深圳地区的补贴", "regional_policy"),
    ]
    
    texts = [item[0] for item in intent_data]
    intents = [item[1] for item in intent_data]
    
    # 生成实体标签
    entities = []
    for text in texts:
        entity_tags = ['O'] * len(text)
        
        # 简单规则匹配实体
        entity_keywords = {
            '汽车': 'B-PRODUCT',
            '家电': 'B-PRODUCT', 
            '手机': 'B-PRODUCT',
            '数码': 'B-PRODUCT',
            '新能源': 'B-PRODUCT',
            '补贴': 'B-SUBSIDY',
            '北京': 'B-LOCATION',
            '上海': 'B-LOCATION',
            '广州': 'B-LOCATION',
            '深圳': 'B-LOCATION',
            '材料': 'B-DOCUMENT',
            '证件': 'B-DOCUMENT',
            '文件': 'B-DOCUMENT',
            '条件': 'B-CONDITION',
            '资格': 'B-CONDITION',
        }
        
        for keyword, label in entity_keywords.items():
            if keyword in text:
                idx = text.index(keyword)
                entity_tags[idx] = label
                # 标记后续字符
                for i in range(idx + 1, min(idx + len(keyword), len(text))):
                    if i < len(entity_tags):
                        entity_tags[i] = label.replace('B-', 'I-')
        
        entities.append(entity_tags)
    
    return texts, intents, entities

# 推理类
class PolicyInference:
    def __init__(self, model_path=None):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
        
        # 加载模型
        self.model = PolicyMultiTaskModel()
        if model_path and torch.cuda.is_available():
            checkpoint = torch.load(model_path, map_location='cuda')
        elif model_path:
            checkpoint = torch.load(model_path, map_location='cpu')
        else:
            checkpoint = None
            
        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射
        self.intent_labels = ['query_subsidy', 'application_process', 'product_scope',
                             'qualification_check', 'document_requirements', 'deadline_query',
                             'regional_policy', 'appeal_process', 'policy_comparison', 'other']
        self.entity_labels = ['O', 'B-SUBSIDY', 'I-SUBSIDY', 'B-PRODUCT', 'I-PRODUCT',
                             'B-LOCATION', 'I-LOCATION', 'B-TIME', 'I-TIME',
                             'B-CONDITION', 'I-CONDITION', 'B-DOCUMENT', 'I-DOCUMENT']
    
    def predict(self, text):
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取意图预测
        intent_logits = outputs['intent_logits']
        intent_pred = torch.argmax(intent_logits, dim=1)
        intent_label = self.intent_labels[intent_pred.item()]
        confidence = torch.softmax(intent_logits, dim=1).max().item()
        
        # 获取实体预测
        entity_logits = outputs['entity_logits']
        entity_preds = torch.argmax(entity_logits, dim=2)
        
        # 提取实体
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        for i, (token, pred_idx) in enumerate(zip(tokens, entity_preds[0])):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            entity_label = self.entity_labels[pred_idx.item()]
            if entity_label != 'O':
                entities.append({
                    'word': token,
                    'entity': entity_label,
                    'position': i
                })
        
        return {
            'text': text,
            'intent': intent_label,
            'confidence': confidence,
            'entities': entities
        }

def main():
    # 检查GPU状态
    print("=== 设备信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print("================")
    
    # 生成训练数据
    print("生成训练数据...")
    texts, intents, entities = generate_policy_data()
    print(f"生成 {len(texts)} 条训练数据")
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    
    # 创建数据集和数据加载器
    train_dataset = PolicyDataset(texts, intents, entities, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    # 创建验证集（使用部分训练数据）
    val_size = min(8, len(texts) // 4)
    val_dataset = PolicyDataset(texts[:val_size], intents[:val_size], entities[:val_size], tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # 初始化模型
    model = PolicyMultiTaskModel()
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总共 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 开始训练
    trainer = PolicyTrainer(model, train_loader, val_loader)
    trainer.train(epochs=3)
    
    # 测试训练好的模型
    print("\n=== 测试模型 ===")
    inference = PolicyInference('best_policy_model.pth')
    
    test_questions = [
        "汽车以旧换新补贴多少钱？",
        "怎么申请家电补贴？",
        "哪些手机可以参与活动？",
        "北京地区的补贴政策是什么？",
        "申请需要什么材料？"
    ]
    
    for question in test_questions:
        result = inference.predict(question)
        print(f"\n问题: {result['text']}")
        print(f"意图: {result['intent']} (置信度: {result['confidence']:.3f})")
        if result['entities']:
            print(f"实体: {[entity['word'] for entity in result['entities']]}")
        else:
            print("实体: 无")

if __name__ == "__main__":
    main()
    print("\n训练完成！最佳模型已保存为 'best_policy_model.pth'")