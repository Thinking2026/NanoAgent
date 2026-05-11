## 任务
基于某LLM API编写一个完整的AI Agent原型应用程序

## 文件说明
- **agent_thread.py**: 负责处理Agent event loop
- **llm_api**: 封装大模型API的调用，以及返回内容的解析
- **main.py**: 主函数实现，核心功能是创建和初始化Agent Thread
- **context/formatter.py**: 负责所有处理所有LLM输入输出信息的格式化和标准化
- **message_queue.py**: Agent线程和用户线程交互的双向队列封装
- **rag_service**: 负责调用外部数据源，将外部数据融合进Prompt的处理类封装
- **agent_context.py**: Agent上下文信息存储，比如包含不断追加的Prompt上下文
- **storage.py**: RAG依赖的外部存储，比如数据库API的封装
- **tools.py**: AI需要调用的工具的标准实现
- **user_thread.py**: 负责处理用户输入消息，给用户输出消息的事件循环
- 数据库操作的安全问题

## 代码扩展点
- llm_api这里需要有一个继承体系，制定标准协议，允许动态支持不同大模型的API
- tools这里需要一个继承体系，制定标准协议，允许动态扩展不同的工具；同时tools里面需要实现一个责任链设计模式，调用端发起一次调用可以路由到特定的工具执行
- storage这里需要一个继承体系，制定标准协议，允许扩展不同的存储，比如SQLLite，向量数据库

## 核心流程设计
### user_thread工作流程
**Step1:** 检查会话状态，如果状态是NEW_TASK，在屏幕输出引导词“Can I Help You ?”, 否则输出引导词"To better solve the problem, you can provide the AI with solution prompts"
**Step2:** 等待用户输入
**Step3:** 将用户的输入通过message_queue投递给agent thread
**Step4:** 进行一个while loop，在这个loop里先轮询message_queue里有没有agent thread投递给user thread的消息，如果没有消息向屏幕输出“Solving...”，然后等待5秒再进入下一次循环；如果发现queue中有agent_thread给user thread投递的消息，将这个消息显示在屏幕上，然后跳出循环
**Step6:** 回到Step1

### agent_thread工作流程
**Step1** 进入一个while True循环
**Step2:** 如果会话状态等于IN_PROGRESS且超过_max_react_attempt_iterations，通过message_queue向user_thread投递一个消息role=System, context="Sorry, this question is too hard, i can not solve"的ChatMessage. 执行成员函数cleanup, continue掉后面处理流程回到循环开始
**Step3:** 从message queue中获取user thread投递过来的用户输入。如果会话状态等于NEW_TASK，且用户还没输入消息需要无限等待，否则最多等待5秒继续执行后面流程
**Step4:**如果会话状态等于NEW_TASK则调用_generate_react_prompt生成一个起始Prompt, _generate_react_prompt这个方法我后面补充
**Step5:**调用message formatter格式化输入，具体如何处理后面再补充
**Step6:**将目前得到的Prompt追加到AgentContext对应字段中
**Step7:**调用LLM API并有限等待返回结果， 如果调用超时，我还没想好怎么处理，先保留一个超时处理策略函数调用
**Step8:**AgentThread需要增加一个成员方法，解析 LLM API的调用返回
**Step9:**如果LLM API返回的信息不是预期格式，也需要加一个兜底处理函数，目前没想好;如果能解析进行行为路由, 先设计三种可能的路由情况：
（1）是一次工具调用：路由给指定工具执行调用并获取工具执行的返回结果
（2）需要查询外部数据库获取信息：调用外部数据源API获取信息
（3）是最终结论：标准化一个chatmessage

如果是（1）和（2）需要先调用message formatter标准化和格式化信息，追加到AgentContext的prompt上下文里 （3）不需要

**Step10:** 如果已经得到最终答案，将答案投递user_thread，并调用cleanup重重会话状态和上下文；

**Step11:** 进入下一轮AgentThread循环



## ReAct Prompt模板思路：
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


## 思考点
- 上下文如何裁剪，token aware怎么做
- 大模型喜欢输入列表，conversation history和system prompt边界
- tool calling的输入和输出标准协议
- 用conversation history统一所有外部调用过程
- 引入checkpoint机制
- 通过缓存直接返回答案
- 将Agent框架分为能力层，韧性层和约束层
- 设计策略和顺序：1. 基础能力与链路 2.可靠性设计 3. 权限，能力边界与审计 4. token aware与节约 5. 任务处理结论的置信度 6. 越用约智能的正反馈模式
  
## TODO List
- [TODO]目前的实现感觉只适合一个任务，如果做成个人住手，要处理多会话，以及上下文统一入口
- [Next]边界情况各种兜底
- [TODO]什么情况下使用RAG，是否可以让大模型自己决定
- [TODO]已经回答过问题的答案，是否要用多级存储
- [TODO]checkpoint机制
- [TODO]Agent执行的审计，需要单独当成一个领域来处理
- [DONE]流程防止无限处理
- [DONE]给LLM传递消息的strip处理要精确
- [DONE]检查工具，环境权限等问题
- [TODO]Agent需要用户协助时要主动提出（比如开权限），等待用户完成后继续工作
- [TODO]多会话，每会话多轮次处理
- [TODO]多级存储怎么做
- [TODO]trace优化，发现有一些冗余不合理的东西，不确定是打印问题还是流程问题
- [TODO]存储里有多个库和表该如何统筹
- [Done]失败返回引发session reset，引发的整个session流程要看一下
- [Done]线程之间的交互方式，每个子线程都可以抛出异常，但抛出异常就意味着无法容忍的错误，子线程要捕获，然后通知所有人清理资源和退出；UserThread开始一个New Task，然后等待agent thread将状态设置为in progress并sync回User Thread；如果agent thread能解决掉需要回传user thread任务完成，user thread打印信息并退出；如果agent thread无法解决任务走异常路径，让全部线程退出
- [Done]把conversion的role去掉
- [Done]喂给LLM API的conversation使用摘要技术，保存system prompt, 用户目标和前面的摘要
- [Done]正反馈路径，类似问题的处理存到RAG里
- [TODO]工具不能每次全量，考虑工具过滤，工具选择等技术方案
- [Done]同一个模型provider都可以使用不同的model，比如claude的sonnet降级到haiku
- [Done]工具注册前要全部自检，除了权限问题确保都是可以执行的
- [TODO]轨迹重放的debug能力
- [TODO]重复的推理轨迹以及工具调用轨迹，系统介入防止死循环
- [Done]对于LLM返回finish reason=length时处理的不好
- [Done]需要根据不同provider回复处理个性化错误码和message
- [TODO]工具调用参数校验，返回格式不对的话先程序修复，修复不了让LLM重新处理
- [Done]上下文剪裁，初步完成
- [TODO]系统迭代次数最后一轮强行给一个结果？
- [TODO]现在Agent执行的任务类型和ReAct绑死了，需要解耦
- [TODO]RAG做内部知识源，让LLM自己选择是否使用
- [TODO]如何路由不同的模型，任务意图识别，cost model...
- [TODO]工具注册那里没有提供显示的注册能力，数据库工具注册那里恶心需要统一化
- [Done]两个问题：（1）Strategy A触发说msg减去1，但是StrageC/D打印msg减2 （2）裁剪器里fits函数重构统一
- [TODO]策略插入修改prompt，比如重复推理单元多，自修复能力等插值
- [TODO]ReAct结合CoT和Plan-and-Execute必须要做
- [TODO]在agent runtime上构建管道式任务
- [TODO]主动智能，不要等用户来用，而是先一步帮他完成；context情景智能，收集信息，必要时使用；无感交互，不改变用户行为习惯更好解决问题
- [TODO]Agent执行完一个工具调用需要更新memory?
- [TODO]先不搞多用户了，未来再说
- [TODO]先不搞推理模式动态切换，未来再说
- [TODO]分析任务时输出工具列表和置信度，然后后续执行的system_prompt里就用这些工具列表
- [TODO]模型选择器里加入熔断器能力，如何决策当前step使用的LLM Provider，每个provider有一个cool off time。连续失败的话cool off time就要变长

## ReAct Agent裁剪上下文设计
- 根据token预算触发裁剪
- [Done]分槽位，每个槽位分配token预算，系统提示15%，多伦推理35%，工具调用和RAG 35%，用户输入15%
- 引用
- 重要性评分
- 时间衰减
- 滑动窗口+summary
- json保留关键结构
- LLM返回的thought可能没用
- [Done]summary也要预留token
- 渐进式裁剪，不要上来就压缩
- 保障LLM API需要的tool call/tool result配对

### token裁剪触发时机
- build-time hard truncate（必备）
- append-time soft truncate（70% 阈值）
- tool-output 强制压缩（必须）s
- step-level summary（ReAct 必备）

## 实践过程中的问题
### 问题
- LLM不调用我提供的工具: DeepSeek习惯使用Python脚本直接处理问题
- 结果文件没有写入：DeepSeek说结果文件已经存到了磁盘，但是实际没有

### 解决方式
- 最开始问问题时就将工具描述带进去，让LLM优先使用我们提供的工具。上面两个问题得到解决