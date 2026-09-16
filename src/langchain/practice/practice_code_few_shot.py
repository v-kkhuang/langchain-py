import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
llm = ChatOpenAI(
    model=os.environ["DEEPSEEK_MODEL"],  # 真实模型名
    base_url=os.environ["DEEPSEEK_API_URL"],
    api_key=os.environ["DEEPSEEK_API_KEY"],  # 手动传 key（因为变量名不是 OPENAI_API_KEY）
)




user_prompt =  """
你是一位拥有 15 年经验的高级软件工程师和安全专家。
请先逐步分析代码逻辑，再给出审查结论。

--- 示例 1：Bug 审查 ---
代码：
def divide(a, b):
    return a / b

审查：
## 逐步分析
1. 函数接收 a 和 b 两个参数
2. 直接执行 a / b，无任何检查
3. 边界情况：b=0 → ZeroDivisionError；a,b 为非数字 → TypeError

## 审查结论
### 🔴 严重 — 除以零风险
- 问题：b 为 0 时程序崩溃
- 修复：
  def divide(a: float, b: float) -> float:
      if b == 0:
          raise ValueError("除数不能为 0")
      return a / b

### 🔵 建议 — 缺少类型标注
- 修复：已在上面的修复代码中添加类型标注

--- 示例 2：安全审查 ---
代码：
def login(username, password):
    user = db.query(f"SELECT * FROM users WHERE name='{username}'")
    if user and user.password == password:
        return create_token(user.id)
    return None

审查：
## 逐步分析
1. username 被直接嵌入 f-string SQL 语句 → SQL 注入
2. 密码用 == 明文比较 → 应该用哈希
3. 无登录失败次数限制 → 暴力破解风险
4. token 创建后无异常处理

## 审查结论
### 🔴 严重 — SQL 注入
- 修复：使用参数化查询
  user = db.query("SELECT * FROM users WHERE name=?", (username,))

### 🔴 严重 — 密码明文比较
- 修复：使用 bcrypt
  if bcrypt.checkpw(password.encode(), user.password_hash):

### 🟡 警告 — 无暴力破解防护
- 修复：添加失败计数和锁定机制

--- 请审查以下代码 ---
代码：
def process_items(items):
    result = []
    for item in items:
        if item > 10:
            result.append(item * 2)
    return result

"""


response = llm.invoke(
     [
            {"role": "user", "content": user_prompt},
        ]
)
print(response.content)

