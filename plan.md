## 任务
编写一个AI Agent的原型代码

## 步骤
### Step1

编写几个类：
- Tool基类：规定好输入和输出的基本格式，统一的方法用于Tool的描述；具体的Tool都要继承这个类，实现具体工具功能方法

内部Tool：ContextTool 更正上下文

- LLM类：负责与大模型API打交道，将Prompt输入给大模型，获取返回

- Parser类：

LLMOutputParser: 负责识别大模型的输出 a. 需要调用工具 b 是其中的思考过程
ToolOutputParser: 负责处理工具调用返回 

AgentContextManager: 记录一次执行的上下文

Agent接口：用来定义基本方法

ReActAgent：定义Agent Loop处理主循环, 需要负责产生Prompt， 带状态 
ReActPromptProducer
a. 等待用户输入问题 b 求解问题 

UserSession：处理用户输入的

MessageQueue: UserSession和Agent互传信息

1. 接收用户问题
2. 加工Prompt
3. 追加到Context（可能每步都要追加）
4. 调用LLM
5. 调用LLMOutputParser解析返回
   4.1 如果是工具调用 路由给对应的工具 执行调用（工具调用没返回怎么处理，工具调用超时）
   4.2 如果是一次思考 只追加context 继续还是调整策略，调整策略可能要调用内部Tool ContextTool修改上下文
   4.3 是最后结论 停止循环，输出结果

   注意 解析不出来 怎么处理

6. 处理用户加入的conversation提示
   5.1 让LLM决定是否需要引入用户帮忙
   5.2 用户session需要与Agent Session隔离 AgentSession需要异步检查是否有用户主动输入的东西

这里先不实现
有限循环 ReAct -> Cot
Cot -> ReAct


ReAct Prompt模板思路：
- **任务分解**：把大问题拆解成小步骤。
  - *例子*：“我需要先搜索x，然后找到y，最后再找z。”
- **信息提取**：从观察到的内容中抓取关键信息。
  - *例子*：“从维基百科看到，x始建于1844年。”
- **常识与算术推理**：运用外部知识或计算。
  - *例子*：“1844 < 1989”（算术比较）或者 “x不是y，所以z应该是...”（逻辑常识）。
- **指导搜索调整**：当搜索不到时，思考如何改变策略。
  - *例子*：“刚才那个没搜到，也许我可以尝试搜索x来代替。”
- **整合最终答案**：把所有信息汇总，得出结论。
  - *例子*：“...所以最终的答案是x




