# 需求预研

## Terms

| #    | 术语               | 解释                                                         |
| ---- | ------------------ | ------------------------------------------------------------ |
| 1    | Task               | 用户提交的任务                                               |
| 2    | Plan               | 任务的整体执行计划                                           |
| 3    | Step               | Plan里制定的任务处理步骤，一个Plan对应N个Step，简单的Plan可以只有一个Step。每个Step都会描述“这一步的目标是什么”/"这一步整体的处理思路"/"这一步需要输出什么关键结果"。 |
| 4    | Stage              | 任务的执行过程中，根据Step创建的执行步骤。Stage带状态轮转(RUNNING/PAUSED/COMPLETED(只是完成，不一定评审通过)/SUCCESS/FAILED)，而Step是静态的对象 |
| 5    | Checkpoint         | 任务执行保存点，当几个Stage成功完成后，Agent可以保存检查点（保存时机看策略）。用户可以要求某个任务从某个Checkpoint开始重新执行 |
| 6    | ER                 | Evaludation Report, 评测报告。Plan/Task的执行结果/Stage的执行结果都会被评测。ER包含的基本信息有评测对象/评测是否通过/评测意见 |
| 7    | User Preference    | 用户偏好                                                     |
| 8    | Knowledge          | 之前任务执行留下来的经验总结                                 |
| 9    | Reasoning Strategy | 特指执行某个Stage时候采用的推理模式，目前只有ReAct：即Thought -> Action -> Observation |
| 10   | 推理框架           | 是Agent整体如何处理任务的模式，目前采用的模式是：1. 任务特征分析 2. 任务分解Stage制定 3. 对每个分解后的Stage运行N轮ReAct |

## User Case

| #    | 用户行为                                       | 行为描述                                                     |
| ---- | ---------------------------------------------- | ------------------------------------------------------------ |
| 1    | 提交任务（用户随时主动发出）                   | 用户向Agent提交一个任务，Agent进行分析和推理，必要时可调用工具与外部交互，最后输出任务处理结论给用户 |
| 2    | 取消任务（用户随时主动发出）                   | 用户向Agent发出取消任务请求，取消的任务不能再继续处理        |
| 3    | 提交建议（用户随时主动发出）                   | 在任务执行期间，如果用户发现执行路径偏离目标可以**主动**提供指导建议，Agent收到用户消息后：1.立即**中止**当前Stage的处理 2.将这一Stage涉及的Context上下文清除掉 3. 结合用户的指导意见，只重新规划当前Stage对应的Step |
| 4    | 用户澄清（系统提示后操作）                     | 如果Agent在执行某步推理时发现需要用户确认，可以主动要求用户澄清，用户提交澄清信息后，Agent继续当前的处理 |
| 5    | 用户要求继续处理（系统提示后操作）             | Agent本质是尽力完成任务的，但可能因为一些原因任务执行出现意外，当某个任务遇到“等待一段时间可以恢复”类的异常中断点，Agent会暂停当前步骤的处理等待用户介入，用户发现异常中断已解决，可以发起继续任务指令，Agent从当前步骤继续执行 |
| 6    | 用户要求从最近检查点重新执行（系统提示后操作） | 因为进程崩溃等异常中断，用户可以要求Agent从指定checkpoint恢复并执行 |

## Agent能力

1. 基于LLM，Tools和Memory能力解决用户提交的任务
2. 当某个执行步骤成功完成，Agent可以选择保存checkpoint，保存checkpoint是异步的，不影响主流程。当无法处理的中断出现，Agent可以restore checkpoint继续执行
3. Agent在每个Stage执行完毕后，需要评估这个Stage执行是否达成Stage目标，没有达到的话revise当前stage的执行计划，然后开始重新执行该Stage 
4. Agent需要对最后任务的结果使用进行评测，评测通过才能交付给用户，否则需要结合评测报告+原执行计划更新整个执行计划，从第一个Stage开始执行
5. Agent暂时实现2个飞轮能力，用户偏好和Task执行总结的经验和知识。用户偏好和Task知识是最后Task成功完成落存储的，未来任务可酌情使用
6. 目前推理框架以及Reasoning Strategy不允许在一个任务执行期间动态调整，留作未来扩展
7. Agent发现一些情况（比如Token不够用）需要等待一段时间才能执行任务时会暂停任务，然后等待用户重新触发继续处理
8. Agent接收到用户的建议后，会立即中断当前步骤处理，结合用户建议只重新规划本Stage对应的计划Step，再从新的Step对应的Stage开始执行，前面已经完成的Stage不受影响
9. Agent执行任务期间，发现需要用户澄清的事实时，可以暂停步骤执行，向用户询问，等待用户澄清后再继续本步骤的处理
10. Agent可以根据任务特征和已支持LLM Provider的能力情况，结合具体路由策略选择合适的LLM Provider

# 执行流程

## 流程三层架构

- Task Level

  ```
  1.开始处理Task
  	1.1 分析Task特征
  		1.1.1 提取任务特征
  		1.1.2 根据任务特征索引用户偏好信息
  		1.1.3 根据任务特征索引知识库
  		1.1.4 发布“分析报告已出”事件（目的是给用户展示执行过程）
  		1.1.5 输出任务分析报告
  	1.2 根据Task特征匹配处理模型，输出可选模型列表
  	1.3 开始制定Task执行计划
  		1.3.1 制定计划
  		1.3.2 评审计划
  			1.3.2.1 [评审成功] 
  				1.3.2.1.1 发布“执行计划已确定”事件（目的是给用户展示执行过程）
  				1.3.2.1.2	交付计划
  			1.3.2.2 [评审不成功] 
  				1.3.2.2.1 [需要用户提供建议] 发布"请求用户建议已发出"事件，阻塞等待在收件箱
  					1.3.2.2.1.1 收到用户建议，注入用户建议，go back to 1.3
  				1.3.2.2.2 [不需要用户提供建议] 结合评审意见 go back to 1.3
  	1.4 发布“Task已开始执行”事件（目的是给用户展示执行过程）
  	1.5 按照计划Step执行计划(进入“Stage Leve“处理，获得处理结果)
  		1.5.1 [执行成功] 对任务结果进行评审
  			1.5.1.1 [评审通过]  
  				1.5.1.1.1 异步提取任务经验和知识+知识落地
  				1.5.1.1.2 从用户建议里总结用户偏好并落地
  				1.5.1.1.3 任务结果交付,发布“Task执行结果信息”事件（目的是给用户展示执行过程）
  			1.5.1.2 [评审不通过] 执行上下文全部清空，结合评审意见重新制定计划，go back to 1.3
  		1.5.2 [执行失败] 任务失败，组装失败信息
  			1.5.2.1 发布“Task执行结果信息”事件（目的是给用户展示执行过程）
  ```

- Stage Level

  ```
  1.从当前Stage开始处理(首次进入从第一个Stage开始)
  	1.0 发布“XX Stage执行开始”事件，标注开始类型[A.新Stage执行 B.Stage执行结果评审不通过，更新Step后重新执行 C.切换模型后重新执行 D.执行失败，更新计划后重新执行]（目的是给用户展示执行过程）
  	1.1 根据条件考虑是否切回高优先级模型
  	1.2 执行当前Stage的推理循环(进入“Stage内部推理循环”，获得处理结果)
  	1.2.1 [Stage执行成功] 对Stage执行结果进行评审
  		1.2.1.1 [评审成功] 对这一步执行情况进行总结（Stage总结要求目标和结果完整，过程摘要），更新上下文
  			1.2.1.1.1 非最后一个Stage发布“Stage执行结果已生成”事件（目的是给用户展示执行过程）
  			1.2.1.1.2 根据条件决定是否要落checkpoint（异步进行）
  			1.2.1.1.3 [还有Stage没处理完] go back to 1处理下一个Stage
  			1.2.1.1.4 [Stage都处理完] 交付最终结果
  		1.2.1.1 [评审不成功] reset掉本Stage上下文，结合评审信息重新规划本Step，go back to 1 从更新后的本Stage开始处理
  	1.2.2 [执行失败，需要切模型] reset掉本Stage上下文
  		1.2.2.1切换模型 go back to 1 重新处理本Stage
  	1.2.3 [Stage内部推理返回需要重新规划本步骤] reset掉本Stage上下文，结合评审信息重新规划本Step，go back to 1 从更新后的本Stage开始处理
  	1.2.4 [执行失败，切模型无法解决] 抛异常，代表无法解决
  ```

- Stage内部推理循环

  ```
  1.获取执行上下文
  	1.1 不满足context window要求，执行压缩或者摘要
  2.调用LLM进行推理，获取下一步Decision
  	2.0 发布"LLM回复已生成"事件（目的是给用户展示执行过程）
  	2.1 [是最终结果] 交付结果
  	2.2 [需要继续推理] 更新上下文 go back to 1
  	2.3 [是工具调用] 开始调用工具
  		2.3.0 发布"工具调用已开始"事件（目的是给用户展示执行过程）
  		2.3.1 检查工具权限和入参是否符合要求
  			2.3.1.1 [允许执行] 调用工具
  				2.3.1.1.1 [调用成功] 将成功结果注入上下文，发布"工具调用结果"事件（目的是给用户展示执行过程），go back to 1
  				2.3.1.1.2 [调用不成功] 
  					2.3.1.1.2.1 [是搜索工具] 尝试用本地知识库 
  						2.3.1.1.2.1.1 [成功] go back to 2.3.1.1.1
  					2.3.1.1.2.2 将失败信息注入上下文，发布"工具调用结果"事件（目的是给用户展示执行过程），go back to 1
  		  2.3.1.2 [工具前置检查不通过] 注入系统信息，让LLM切换工具，发布"工具调用结果"事件（目的是给用户展示执行过程），go back to 1
  	2.4 [需要用户澄清信息] 更新必要状态，发布“用户澄清请求已发出”事件，阻塞等待继续处理
  		2.4.1 收到用户澄清信息
  			2.4.1.1 更新上下文，go back to 1
  	2.5 [发现任务需要暂停]，更新必要状态，发布“任务已暂停”事件，阻塞等待继续处理
  		2.5.1 收到继续任务的指令
  			2.5.1.1 go back to 1
  3.Loop用户异步提交的信息
  	3.1 用户要求取消任务
  		3.1.1 更新任务状态，清空相关信息，发布“任务已取消”事件
  	3.2 收到用户主动纠偏建议
  		3.2.1 返回“需要重新规划本步骤”信息给外层
  ```

- UC-6 用户要求从Checkpoint处执行

```
1. 获取Task执行计划和执行进度
2. 重构Agent Context
3. 从“Task Leve”中的1.4步开始执行
```

## 重要聚合实体

| #    | 实体                | 功能语义                                                     |
| ---- | ------------------- | ------------------------------------------------------------ |
| 1    | Analyzer            | 提取任务特征信息，输出任务分析结果（包含可以利用的用户偏好和相关知识片段），有专门的analyise LLM Provider |
| 2    | Planner             | 制定计划，更新整个计划或者更新计划的某个Stage。制定计划时Planner还要负责调用QualityEvaluator评审整个计划。最终输出一个“计划”详情，有专门的planner LLM Provider |
| 3    | CheckpointProcessor | 负责执行点checkpoint的save/restore/list/get/delete，checkpoint有版本和时间信息 |
| 4    | KnowledgeManager    | 负责1.总结任务处理经验和知识 2.存储经验和知识 3.删除无用的经验和知识 |
| 5    | QualityEvaluator    | 负责 1.评估整体执行结果是否满足任务目标 2.评估某个Stage执行是否符合预期目标 3.评审执行计划是否符合满足任务目标 |
| 6    | StageExecutor       | 驱动和执行计划的所有Stage                                    |
| 7    | KnowledgeLoader     | 负责query与任务相关的可能用上的知识                          |
| 8    | ModelSelector       | 负责根据任务特征，模型特点，选择本任务适合的模型和备选模型   |
| 9    | ContextManager      | 负责管理Task执行上下文，包含常规的增删改查，以及为了适应LLM Provider的context_window要求进行裁剪/压缩/摘要，有专门的Context压缩摘要的LLM Provider |
| 10   | ReasoningManager    | 负责与LLM打交道，执行单步推理，并输出Next Decision           |
| 11   | LLMGateway          | 1. 封装LLM不同provider的API 2.处理标准请求/回复协议与各个Provider请求/回复协议的互转处理 3. 一些LLM Provider调用级别的基础容错，比如调用API超时的自动backoff jitter重试 |
| 12   | ToolRegistry        | 1. 封装不同工具的调用和返回 2. 处理标准参数/回复协议与各个Tool参数/回复协议的互转 3. 处理一些Tool调用级别的基础容错，比如调用工具超时的自动backoff jitter重试 |
| 13   | PersonalityManager  | 1. 索引用户偏好信息 2. 提炼用户偏好 3. 存储用户偏好          |

## 应用层实体

| #    | 名称           | 语义                                                         |
| ---- | -------------- | ------------------------------------------------------------ |
| 1    | Pipeline       | 处理Task Level执行流程                                       |
| 2    | PipelineThread | Pipeline执行的容器 1. 负责管理执行线程的生命周期 2. 维护与client端通信的异步队列 |
| 3    | PipelineDriver | pipeline与client沟通的桥梁 1. 负责client报文与Pipeline Command互转 2. 提供pipeline标准事件handler，处理pipeline执行过程想向client端发送的消息 |
| 4    | client_app     | 代表用户端的执行线程，负责与用户UI相关的所有逻辑             |

### 应用层实体的同步设施

#### PipelineThread

- TaskQueue: client_app通过这个队列向PipelineThread发送新Task信息/断点恢复消息
- AgentMessageQueue: client_app通过这个队列向PipelineThread发送用户指令（取消/建议/澄清/resume）

#### client_app

- UserMessageQueue: Pipeline Thread通过这个队列向client端发送Agent执行信息，任务结果信息

### 应用层实体交互关系

- 用户提交任务

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      client->>thread: 通过TaskQueue发送任务
      thread->>driver: 发送任务信息
      driver->>driver: 协议转换
      driver->>pipeline: 调用pipeline任务处理接口
      
  ```

  

- 用户要求从checkpoint恢复执行

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      client->>thread: 通过TaskQueue发送恢复检查点的指令
      thread->>driver: 发送指令
      driver->>driver: 协议转换
      driver->>pipeline: 调用pipeline接口继续处理任务
      
  ```

- 用户取消任务

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      client->>thread: 通过AgentMessageQueue发送取消任务指令
      pipeline->> driver: loop用户指令
      driver->>thread: fetch用户指令
      thread->>driver:返回用户指令
      driver->>driver:协议转换
      driver->>pipeline:返回取消任务指令
      pipeline->>pipeline:更新任务状态/清空上下文
      pipeline->>driver:发布"任务已取消"事件
      driver->>driver:1.接收事件 2.协议转换
      driver->>thread:发送用户信息
      thread->>client:传递用户信息
      thread->>thread:阻塞在TaskQueue等待新任务或者继续执行某任务
  ```

- 用户向Agent提交执行建议

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      client->>thread: 通过AgentMessageQueue发送建议
      pipeline->> driver: loop用户指令
      driver->>thread: fetch用户指令
      thread->>driver:返回用户指令
      driver->>driver:协议转换
      driver->>pipeline:返回建议提交指令
      pipeline->>pipeline:走Stage处理用户建议步骤及后续步骤......
  ```

- Pipeline需要用户提供澄清信息

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      pipeline->>driver:发布需要用户澄清事件
      driver->>driver:协议转换
      driver->>thread:给用户发送信息
      thread->>client:通过UserMessageQueue给用户发送信息
      thread->>driver:返回
      driver->>pipeline:返回
      pipeline->> driver: loop用户指令
      driver->>thread: fetch用户指令
      client->>thread: 通过AgentMessageQueue发送用户澄清信息
      thread->>driver:返回用户信息
      driver->>driver:协议转换
      driver->>pipeline:提交用户澄清信息
      pipeline->>pipeline:走Stage处理用户澄清信息步骤及后续步骤......
  ```

- pipeline暂停处理+用户要求继续处理

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      pipeline->>pipeline: 更新任务状态
      pipeline->>driver:发布任务已暂停事件
      driver->>driver:协议转换
      driver->>thread:给用户发送信息
      thread->>client:通过UserMessageQueue给用户发送信息
      thread->>driver:返回
      driver->>pipeline:返回
      pipeline->> driver: loop用户指令
      driver->>thread: fetch用户指令
      client->>thread: 通过AgentMessageQueue发送用户要求继续的指令
      thread->>driver:返回用户信息
      driver->>driver:协议转换
      driver->>pipeline:提交用户指令
      pipeline->>pipeline:走Stage处理暂停恢复步骤及后续步骤......
  ```

  

- Pipeline向用户提供执行流程信息

  ```mermaid
  sequenceDiagram
      participant client as client_app
      participant thread as pipeline_thread
      participant driver as pipeline_driver
      participant pipeline 
      pipeline->>driver:发布事件
      driver->>driver:协议转换
      driver->>thread:给用户发送信息
      thread->>client:通过UserMessageQueue给用户发送信息
      thread->>driver:返回
      driver->>pipeline:返回
      pipeline->> pipeline: 继续后续流程....
  ```

  

## 代码目录结构


```
NanoAgent/
├── bin/                          # 二进制目录
│   └── nanoagent
├── config/                       # 运行时配置文件
│   └── config.json
├── docs/                         # 设计文档
│   ├── TD.md                     # 本技术设计文档
│   ├── plan.md
│   ├── archive/                  # 历史设计文档归档
│   └── knowledge/                # 知识库文档
├── src/
│   ├── main.py                   # 程序主函数
│   ├── agent/                    # 核心 Agent 领域
│   │   ├── application/
│   │   │   └── pipeline.py         # 应用层编排（Pipeline）
│   │   │   └── driver.py           # pipeline driver
│   │   │   └── pipeline_thread.py  # pipeline thread
│   │   ├── events/               # 领域事件定义
│   │   ├── factory/
│   │   └── models/               # 领域模型
│   │   		├── analysis/         # Analyzer聚合
│   │       ├── checkpoint/       # CheckpointProcessor 聚合
│   │       ├── context/          # ContextManager 实体
│   │       │   ├── budget/       # Token 预算管理
│   │       │   ├── estimator/    # Token 估算
│   │       │   └── truncation/   # 上下文裁剪策略
│   │       ├── evaluate/         # QualityEvaluator 聚合
│   │       ├── executor/         # StageExecutor 聚合
│   │       ├── knowledge/        # KnowledgeManager 聚合 + KnowledgeLoader聚合
│   │       ├── model_routing/    # ModelSelector聚合
│   │       ├── personality/      # PersonalityManager聚合
│   │       ├── plan/             # Planner 聚合
│   │       └── reasoning/        # ReasoningManager（Strategy 抽象 + ReAct 实现）
│   │           └── impl/react/
│   ├── config/                   # 配置处理相关
│   │   ├── config.py
│   │   └── reader.py
│   ├── driver/                   # 用户线程
│   │   ├── demo.py
│   │   └── user_thread.py
│   ├── infra/                    # 基础设施
│   │   ├── cache/                # 负责缓存相关
│   │   ├── db/                   # 存储后端（SQLite/MySQL/ChromaDB）
│   │   │   ├── storage.py        # 存储抽象接口
│   │   │   ├── registry.py       # StorageRegistry
│   │   │   └── impl/
│   │   ├── eventbus/             # 事件总线
│   │   │   └── event_bus.py
│   │   └── observability/        # 可观测性（Metrics/Tracing）
│   ├── llm/                      # LLM 网关层
│   │   ├── llm_gateway.py        # LLMGateway聚合
│   │   ├── registry.py           # LLMProviderRegistry
│   │   ├── providers/            # 各 Provider 实现
│   │   │   ├── claude_api.py
│   │   │   ├── openai_api.py
│   │   │   ├── qwen_api.py
│   │   │   ├── kimi_api.py
│   │   │   ├── minmax_api.py
│   │   │   ├── glm_api.py
│   │   │   └── deepseek_api.py
│   ├── schemas/                  # 跨层共享类型
│   │   ├── types.py              
│   │   ├── errors.py            
│   │   ├── consts.py
│   │   ├── event_bus.py
│   │   ├── ids.py
│   │   └── message_convert.py
│   │   └── interface.py 					#定义基类
│   │   └── task.py 					    #定义一些和Task相关的对象
│   ├── tools/                    # 工具层
│   │   ├── tool_registry.py      # ToolRegistry / ToolChainRouter
│   │   ├── tool_base.py          # 工具基类
│   │   └── impl/                 # 工具实现
│   │       ├── search_tool.py
│   │       ├── sql_query_tool.py
│   │       ├── sql_schema_tool.py
│   │       ├── vector_search_tool.py
│   │       ├── vector_schema_tool.py
│   │       ├── shell_tool.py
│   │       ├── file_tool.py
│   │       ├── excel_tool.py
│   │       ├── calculator_tool.py
│   │       ├── current_time_tool.py
│   │       └── run_python_tool.py
│   └── utils/                    # 通用功能函数
│       ├── concurrency/          
│       ├── env_util/
│       ├── http/
│       ├── log/
│       └── time/
└── tests/
    ├── unit/
    ├── integration/
    └── runtime/
```

## 关键类定义

### Task

- 文件位置：`src/schemas/task.py`
- 职责：没有方法的静态对象，提供围绕Task的基础信息   

#### 任务难度定义

| 复杂度      | 特征                     | 适用场景                         |
| ----------- | ------------------------ | -------------------------------- |
| **L1 简单** | 单步、模板化、低幻觉要求 | 寒暄、格式化、标签分类、简单提取 |
| **L2 标准** | 单步推理、常识、短上下文 | 客服问答、邮件起草、基础翻译     |
| **L3 复杂** | 多步推理、代码、分析     | 代码审查、数据分析、报告生成     |
| **L4 专家** | 深度推理、创意、长链思维 | 架构设计、数学证明、策略规划     |

#### 成员变量

- id: task 唯一ID
- user_id: 提交任务的用户唯一ID
- description: 用户提交的原始任务描述
- task_type: 任务类型标签，如 "data_analysis", "code_generation"
- complexity: "L1简单" | "L2标准" | "L3负责" | "L4专家"
- required_tools:      预估需要的工具名称列表
- reasoning_depth:     单步推理 | 多步推理
- output_constraints: 输出约束，包括格式要求/长度要求/实效性要求/语言要求等等
- notes: LLM 分析备注（约束、风险、前提条件等）
- related_user_preference_entries: 任务相关用户偏好，包含两个部分：一是user_preference信息 二是置信度0-1
- related_knowledge_entries: 任务相关的先验知识，包含两个部分：一是user_preference信息 二是置信度0-1
- plan_id: task关联的执行计划ID

### PlanStep

- 文件位置：`src/schemas/task.py`
- 职责：没有方法的静态对象，描述任务计划一个步骤的基础信息

#### 成员变量

- order: step步骤编号
- goal: 该步骤的目标
- description: 该步骤做什么的描述
- key_results: list[str]. 这一步要产生哪些关键输出

### Plan

- 文件位置：`src/schemas/task.py`
- 职责：没有方法的静态对象，描述任务的执行计划

#### 成员变量

- id: plan 唯一ID
- task_id: 关联的task id
- step_count: 执行计划的步骤数
- step_list: list[PlanStep]. 步骤列表
- created_at

### PlanVersion

- 文件位置：`src/schemas/task.py`
- 职责：描述历史版本的执行计划

#### 成员变量

- plan: Plan
- change_reason  #计划评审不通过/任务结果评审不通过/任务步骤结果评审不通过/用户主动纠偏/

### EvaluationResult

- 文件位置：`src/schemas/task.py`
- 职责：描述评测结果

#### 成员变量

- evaluation_target: 评测对象 plan | task_result | stage_ result
- passed: 是否通过
- feedback: 评测反馈

### User Preference Entry

- 文件位置：`src/schemas/types.py`
- 职责：用户偏好

#### 成员变量

- user_id: 用户ID
- keyword: 偏好关键字
- content: 偏好内容

### Knowledge Entry

- 文件位置：`src/schemas/types.py`
- 职责：知识条目

#### 成员变量

- entry_id: 用户ID
- title: 标题
- tags: 标签列表
- content: 知识内容

### Analyzer

#### 文件位置

`src/agent/models/plan/analyzer.py`

#### 职责

1. 调用LLM或者策略类提取任务特征信息
2. 根据特征信息获取相关用户偏好信息
3. 根据特征信息获取相关knowledge信息
4. 正常输出Task对象，异常抛异常

#### 接口

**接口名**

analyze(self, task_description, llm_gateway, knowledge_loader, personality_manager, tool_registry) ->Task

**输入**

- 用户提交的原始任务信息
- llm_gateway: 调用LLM Provider
- knowledge_loader: 获取任务相关知识
- personality_manager: 获取用户偏好信息
- tool_registry: 工具列表

**输出**

Task对象

### CheckpointProcessor（聚合根）

#### 文件位置

`src/agent/models/checkpoint/checkpoint_processor.py`

#### 职责

1. 为了未来能恢复执行，要把所有关键信息：agent context，当前计划，当前执行进度等等整理成一份checkpoint
2. 存储checkpoint
3. 能索引指定任务的checkpoint
4. list某个任务各个版本的checkpoint
5. 删除某个任务某个版本的checkpoint
6. 选择任务的某个版本checkpoint恢复agent context，当前执行计划，运行时信息比如当前执行进度等

#### 成员变量

需要定义

#### 方法

需要定义

#### 关键输出

checkpoint聚合实体，需要定义

### ContextManager

#### 文件位置

`src/agent/models/context/manager.py`

#### 职责

1. 管理Agent执行上下文的增删改查
2. 调用LLM API时能获取当前的上下文
3. 发现当前上下文超出token限制，需要执行裁剪，压缩，摘要等操作
4. 按照不同的Stage管理上下文，可以清除掉某个Stage涉及的上下文

#### 成员变量

需要定义，至少包含的语义信息：(1) system prompt (2) tool schema描述 （3）动态注入的用户偏好 （4）和Stage相关的用户主动发送的纠偏建议，纠偏建议可以未来用来提取用户偏好，要能单独识别出来（5）注入的相关knowledge （6）和LLM API交互过程中产生的带role的message信息 (7) 需要知道每个stage涉及哪些带role的message (8) 当前已成功完成的最大stage编号 (9) 关联的原始用户任务描述 （10）关联的任务分析报告 （11）关联的任务执行计划信息 （12）必要的元信息

#### 特殊要求

ContextManager的管理的上下文信息，先要转换成标准协议里的LLMMessage，再由各个LLM Provider转成自己的LLMRequest

#### 方法签名

需要定义

#### 设计说明

- tool call 配对修复：如果消息列表末尾存在 `tool_use` 但没有对应的 `tool_result`，`get_context_window()` 会移除该孤立的 `tool_use`，防止 LLM API 报错
- `summarize()` 的 `SummarizationStrategy` 是抽象接口，当前实现为 LLM 摘要（调用 LLMGateway），未来可替换为规则摘要
- context manager需要与其他聚合配合，调用TokenEstimator估算上下文消耗的token数，调用TokenBudgetManager获取预算分配，调用ContextTruncator进行上下文裁剪，压缩，抽摘要

### QualityEvaluator

#### 文件

`src/agent/models/evaluate/quality_evaluator.py`

#### 职责

1. 评估整体任务结果是否符合预期
2. 评估单个Stage执行结果是否符合预期
3. 评估执行计划设计是否符合预期

#### 成员变量

- 无

#### 接口

evaluate_plan(task: Task, plan:Plan, llmgateway: LLMGateway): EvaluationResult

evaluate_task_result(task: Task, result: str, llmgateway: LLMGateway): EvaluationResult

evaluate_stage_result(step: PlanStep, result:str, llmgateway: LLMGateway): EvaluationResult



### StageExecutor

#### 文件

`src/agent/models/executor/stage_executor.py`

#### 职责

1. 负责“Agent执行流”中Stage Level的流程框架
2. 负责“Agent执行流”中Stage内部推理循环的流程框架



#### 实体

需要定义，注意Plan里的Step和这里的Stage是两个对象。Step偏静态，而Stage偏动态，Stage可能挂在动态执行时候的一些信息

#### 枚举：StageStatus

```python
class StageStatus(str, Enum):
    RUNNING     = "RUNNING"      # 推理循环进行中
    COMPLETED   = "COMPLETED"    # 产出最终答案
    PAUSED      = "PAUSED"       # B类异常暂停或等待用户澄清
    FAILED      = "FAILED"       # C类错误或超过最大迭代次数
    等等
```

#### 成员变量        

驱动两层框架循环，需要依赖谁，哪个聚合就成为它的成员

#### 方法签名

至少要两个方法：（1）execute负责Stage level循环 （2）execute_one_stage负责Stage内部推理循环

### Planner

#### 文件位置

`src/agent/models/plan/planner.py`

#### 职责

- 制定计划 
- Plan评审不通过或者Task执行结果不符合预期时，更新整个计划
- 更新计划的某个Stage 
- 调用QualityEvaluator评审整个计划
- 计划制定或者更新过程中，需要用户澄清时，可以发布“需要用户澄清事件”，然后阻塞在消息队列等待消息

#### 成员变量

- current_plan: 当前计划
- history_plan: list[PlanVersion] 历史计划

#### 方法签名

make_plan(task: Task, llm_api: llmgateway, evaluator: QualityEvaluator) : Plan —— 根据Task里面有用的字段，调用大模型llmgateway生成一个plan

renew_plan(task:Task, feedback: str,llm_api: llmgateway): Plan —— 结合feedback, 调用大模型llmgateway重新生成一个plan

renew_plan_step(step:PlanStep, feedback:str, llm_api: llmgateway): PlanStep —— 结合feedback重新制定某个Step



## 标准协议

### LLM Context 协议

#### 数据流

```
ContextManager.get_context_window()
  → ContextWindow { system_prompt, messages: list[LLMMessage], token_count }
      ↓ Strategy.build_llm_request(context_window, tool_registry)
LLMRequest { system_prompt, messages, tools, max_tokens, temperature }
      ↓ LLMGateway.call(request) → Provider.generate(request)
Provider API Request（provider-specific JSON，见序列化规则）
      ↓ HTTP POST
Provider API Response（provider-specific JSON）
      ↓ Provider._parse_response()
LLMResponse { assistant_message, tool_calls, finish_reason, usage }
```

#### 标准类型定义

```python
# src/schemas/types.py

@dataclass(slots=True)
class LLMRequest:
    messages: list[LLMMessage]
    system_prompt: str | None = None
    tools: list[dict[str, Any]] | None = None   # JSON Schema list
    max_tokens: int = 1024
    temperature: float = 0.0                    # 默认确定性输出

@dataclass(slots=True)
class LLMResponse:
    assistant_message: LLMMessage               # role="assistant"
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"                 # "stop" | "tool_use" | "length" | "error"
    usage: LLMUsage | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@dataclass(slots=True)
class LLMMessage:
    role: LLMRole                               # "user" | "assistant" | "tool"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata 约定字段：
    #   tool_calls: list[dict]       — assistant 消息携带工具调用时
    #   llm_raw_tool_call_id: str    — tool 消息关联的调用 ID
    #   tool_name: str               — tool 消息对应的工具名
```

#### LLMMessage.metadata 约定

assistant 消息携带工具调用时，`metadata["tool_calls"]` 格式：

```json
[
  {
    "name": "search",
    "llm_raw_tool_call_id": "toolu_01XYZ",
    "arguments": { "query": "..." }
  }
]
```

tool 消息（工具结果）时，`metadata` 格式：

```json
{
  "llm_raw_tool_call_id": "toolu_01XYZ",
  "tool_name": "search"
}
```

#### Provider 序列化规则

| 字段 | Claude API | OpenAI-compatible API |
|------|-----------|----------------------|
| system_prompt | 顶层 `"system"` 字段 | role="system" 消息插入 messages[0] |
| user/assistant 文本 | `{"role": "user/assistant", "content": "..."}` | 同左 |
| assistant + tool_calls | content 数组含 text block + tool_use block | `{"role": "assistant", "tool_calls": [...]}` |
| tool result | role="user"，content 数组含 tool_result block | `{"role": "tool", "tool_call_id": "...", "content": "..."}` |
| tools schema | `input_schema` 字段 | `parameters` 字段 |
| finish_reason | `stop_reason`: "end_turn" / "tool_use" / "max_tokens" | `finish_reason`: "stop" / "tool_calls" / "length" |

finish_reason 归一化（Provider 原始值 → 标准值）：

```
Claude "end_turn"   → "stop"
Claude "tool_use"   → "tool_use"
Claude "max_tokens" → "length"
OpenAI "stop"       → "stop"
OpenAI "tool_calls" → "tool_use"
OpenAI "length"     → "length"
```

### Tool 协议

#### 完整数据流

```
LLMResponse.tool_calls: list[ToolCall]
  ↓ StageExecutor 发布 ToolCallRequested(E33)（含参数检查、权限检查）
ToolRegistry.execute(tool_call: ToolCall) → ToolResult
  ↓ 发布 ToolCallDispatched(E34)
BaseTool.run(arguments: dict) → ToolResult
  ↓ 成功: 发布 ToolCallSucceeded(E35)
  ↓ 失败: 发布 ToolCallFailed(E36)
ToolResult
  ↓ Strategy.format_tool_observation(tool_call, result) → LLMMessage(role="tool")
ContextManager.add_message(role="tool", content=..., metadata={llm_raw_tool_call_id, tool_name})
  ↓ 发布 ResultInjected(E37)
```

#### 接口定义

```python
# src/tools/models.py
class BaseTool(ABC):
    name: str
    description: str
    parameters: dict[str, Any]   # JSON Schema (type: object)

    @abstractmethod
    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """
        契约：
        - 成功: ToolResult(output=json_str, success=True)
        - 业务失败: ToolResult(output=json_str, success=False, error=AgentError)
          output 仍为合法 JSON（含 error 字段），注入上下文让 LLM 感知
        - 超时: 抛出 AgentError(code=TOOL.A.TIMEOUT)，由 ToolRegistry 捕获重试
        - 不允许抛出其他异常（BaseTool 实现必须内部 catch 并转换为 ToolResult）
        """

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}
```

#### 输出格式（build_tool_output）

所有工具通过 `build_tool_output()` 构造标准 JSON 输出：

成功：
```json
{ "success": true, "data": { ... } }
```

失败：
```json
{
  "success": false,
  "error": {
    "code": "TOOL.A.EXECUTION_ERROR",
    "message": "具体错误描述"
  }
}
```

#### 超时重试契约

`ToolRegistry` 对 `AgentError(code=TOOL.A.TIMEOUT)` 和 `TimeoutError` 自动退避重试：
- 按 `timeout_retry_delays` 序列退避（如 `(1.0, 2.0, 4.0)`）
- 超出重试次数 → 返回 `ToolResult(success=False, error=AgentError(TOOL.C.TIMEOUT_EXHAUSTED))`

---

### EventBus 协议

#### 接口定义

```python
# src/schemas/event_bus.py / src/infra/eventbus/event_bus.py
class EventBus(ABC):
    def publish(self, event: DomainEvent) -> None:
        """同步发布；所有订阅者按注册顺序调用；
        单个 handler 异常不影响其他 handler"""

    def subscribe(self, event_type: type[DomainEvent],
                  handler: Callable[[DomainEvent], None]) -> None

    def unsubscribe(self, event_type: type[DomainEvent],
                    handler: Callable[[DomainEvent], None]) -> None
```

#### DomainEvent 基础结构

```python
# src/schemas/domain.py
@dataclass
class DomainEvent:
    event_type: str        # 事件名，如 "TaskReceived"
    aggregate_id: str      # 聚合根 ID
    occurred_at: datetime  # UTC 时间
    metadata: dict         # 扩展字段（如 task_id, step_id, feedback 等）
```

#### AggregateRoot 事件收集模式

```python
# src/schemas/domain.py
class AggregateRoot(ABC):
    _pending_events: list[DomainEvent]

    def _record(self, event: DomainEvent) -> None:
        """聚合根内部记录事件，不立即发布"""

    def pull_events(self) -> list[DomainEvent]:
        """返回并清空待发布事件列表；由应用层调用后统一发布"""
```

---

### Storage 协议

#### 存储层次结构

```python
# src/infra/db/storage.py
class BaseStorage(ABC):
    backend_name: str

class RelationalStorage(BaseStorage):
    def query(self, request: SQLQueryRequest) -> list[dict[str, Any]]
    def inspect_schema(self, database: str | None = None,
                        table: str | None = None) -> dict[str, Any]

class VectorStorage(BaseStorage):
    def search(self, request: VectorSearchRequest) -> list[dict[str, Any]]
    def inspect_schema(self, collection: str | None = None) -> dict[str, Any]

class KeyValueStorage(BaseStorage):
    def get(self, request: KeyValueGetRequest) -> dict[str, Any] | None
    def set(self, request: KeyValueSetRequest) -> None
    def delete(self, key: str) -> bool

class DocumentStorage(BaseStorage):
    def get_documents(self) -> list[dict[str, Any]]
```

#### StorageRegistry

```python
# src/infra/db/registry.py
class StorageRegistry:
    def register(self, storage: BaseStorage) -> None
    def get(self, backend_name: str) -> BaseStorage
    def list_backends(self) -> list[str]
```

#### 已支持后端

| backend_name | 类型 | 文件 |
|---|---|---|
| sqlite | RelationalStorage | infra/db/impl/sqlite_storage.py |
| mysql | RelationalStorage | infra/db/impl/mysql_storage.py |
| chromadb | VectorStorage | infra/db/impl/chromadb_storage.py |


---

## 错误码体系

### 三维错误码设计

每个 `AgentError` 携带三个维度，Pipeline 只需检查 `recovery` 即可决定处理策略：

| 维度 | 类型 | 说明 |
|------|------|------|
| `business` | BusinessCategory | 错误来源域 |
| `recovery` | RecoveryCategory | Pipeline 恢复策略（对应 A/B/C 类） |
| `code` | str | 具体错误标识，格式 `DOMAIN.RECOVERY.NAME` |

```python
# src/schemas/errors.py

class BusinessCategory(str, Enum):
    LLM     = "LLM"      # LLM API 调用错误
    TOOL    = "TOOL"     # 工具执行错误
    SYSTEM  = "SYSTEM"   # Agent 内部逻辑错误
    STORAGE = "STORAGE"  # 存储层错误
    CONFIG  = "CONFIG"   # 配置错误

class RecoveryCategory(str, Enum):
    A = "A"   # 立即可恢复：修改参数/降级后重试，不暂停任务
    B = "B"   # 等待后可恢复：暂停任务（TaskPaused），等待用户触发继续
    C = "C"   # 不可恢复：终止任务（TaskTerminated）或跳过步骤后重规划

@dataclass
class AgentError(Exception):
    business: BusinessCategory
    recovery: RecoveryCategory
    code: str                      # 完整错误码，如 "LLM.A.TRANSIENT"
    message: str
    cause: Exception | None = None
    retry_after: float | None = None   # B 类错误建议等待时间（秒）
```

---

### LLM 调用错误码

| 错误码 | 触发条件 | Recovery | Pipeline 处理 |
|--------|---------|----------|--------------|
| `LLM.A.TRANSIENT` | 网络错误、5xx、连接超时 | A | LLMGateway 内部退避重试；超出次数 → 切换 provider |
| `LLM.A.RATE_LIMITED` | HTTP 429 | A | LLMGateway 按 retry_after 退避重试；超出次数 → 切换 provider |
| `LLM.A.CONTEXT_TOO_LONG` | HTTP 400 context 超限 | A | StageExecutor 触发 ContextManager.trim_to_max_tokens() 后重试 |
| `LLM.A.RESPONSE_PARSE` | 响应格式无法解析 | A | StageExecutor 触发 self-repair（修正最后一条 assistant 消息）后重试 |
| `LLM.B.OVERLOADED` | 服务过载（如 Claude 529），短期无法恢复 | B | TaskPaused，等待用户触发继续 |
| `LLM.C.AUTH_FAILED` | HTTP 401/403 | C | 跳过当前 provider；所有 provider 失败 → TaskTerminated |
| `LLM.C.CONFIG_ERROR` | API key 缺失/配置错误 | C | 跳过当前 provider；所有 provider 失败 → TaskTerminated |
| `LLM.C.ALL_PROVIDERS_FAILED` | 所有 provider 均失败 | C | TaskTerminated |

---

### Tool 调用错误码

| 错误码 | 触发条件 | Recovery | Pipeline 处理 |
|--------|---------|----------|--------------|
| `TOOL.A.TIMEOUT` | 工具执行超时（单次） | A | ToolRegistry 退避重试 |
| `TOOL.A.EXECUTION_ERROR` | 工具执行失败（业务错误） | A | 错误信息注入上下文，LLM 决策下一步 |
| `TOOL.A.ARGUMENT_ERROR` | 参数校验失败 | A | 错误信息注入上下文，LLM 修正参数重试 |
| `TOOL.B.RESOURCE_UNAVAILABLE` | 外部资源暂时不可用（DB 连接失败等） | B | TaskPaused，等待用户触发继续 |
| `TOOL.C.TIMEOUT_EXHAUSTED` | 超时重试次数耗尽 | C | 注入错误，触发 TaskPlanRevised（跳过或替换步骤） |
| `TOOL.C.NOT_FOUND` | 工具不存在 | C | 注入错误，触发 TaskPlanRevised |
| `TOOL.C.PERMISSION_DENIED` | 权限检查失败 | C | 注入错误，触发 TaskPlanRevised |

---

### 系统内部错误码

| 错误码 | 触发条件 | Recovery | Pipeline 处理 |
|--------|---------|----------|--------------|
| `SYSTEM.A.MAX_ITERATIONS` | 单 Stage 推理轮次超限 | A | 触发 TaskPlanRevised（拆分或简化步骤） |
| `SYSTEM.A.STAGE_INFEASIBLE` | Stage 执行中发现步骤无法完成 | A | 触发 TaskPlanRevised |
| `SYSTEM.B.TOKEN_BUDGET_EXHAUSTED` | Token 预算耗尽，无法继续 | B | TaskPaused，等待用户触发继续 |
| `SYSTEM.C.INTERNAL_ERROR` | 未预期的内部异常 | C | TaskTerminated |
| `SYSTEM.C.MAX_PLAN_RETRIES` | 计划重试次数超限 | C | TaskTerminated |
| `SYSTEM.C.MAX_QUALITY_RETRIES` | 质检重试次数超限 | C | TaskTerminated |

#### Storage 错误码

| 错误码 | 触发条件 | Recovery | Pipeline 处理 |
|--------|---------|----------|--------------|
| `STORAGE.A.QUERY_ERROR` | 查询执行失败 | A | 注入错误，LLM 决策 |
| `STORAGE.B.CONNECTION_FAILED` | 存储连接失败 | B | TaskPaused |
| `STORAGE.C.CONFIG_ERROR` | 存储配置错误 | C | TaskTerminated |

---

### Provider 原始错误 → AgentError 映射

LLMGateway 负责将 provider 抛出的原始异常转换为 `AgentError`，上层只感知 `AgentError`：

```
HTTP 429                          → AgentError(LLM, A, "LLM.A.RATE_LIMITED",   retry_after=...)
HTTP 401/403                      → AgentError(LLM, C, "LLM.C.AUTH_FAILED")
HTTP 400 + context hints          → AgentError(LLM, A, "LLM.A.CONTEXT_TOO_LONG")
HTTP 529（Claude overloaded）     → AgentError(LLM, B, "LLM.B.OVERLOADED")
HTTP 5xx（其他）                  → AgentError(LLM, A, "LLM.A.TRANSIENT")
NetworkError / ConnectionError    → AgentError(LLM, A, "LLM.A.TRANSIENT")
TimeoutError                      → AgentError(LLM, A, "LLM.A.TRANSIENT")
ResponseParseError                → AgentError(LLM, A, "LLM.A.RESPONSE_PARSE")
MissingAPIKey / BadConfig         → AgentError(LLM, C, "LLM.C.CONFIG_ERROR")
```

Tool 层将原始异常转换为 `AgentError`，ToolRegistry 只感知 `AgentError`：

```
TimeoutError                      → AgentError(TOOL, A, "TOOL.A.TIMEOUT")
ValueError（参数校验）             → AgentError(TOOL, A, "TOOL.A.ARGUMENT_ERROR")
PermissionError                   → AgentError(TOOL, C, "TOOL.C.PERMISSION_DENIED")
ConnectionError（外部资源）        → AgentError(TOOL, B, "TOOL.B.RESOURCE_UNAVAILABLE")
Exception（其他）                 → AgentError(TOOL, A, "TOOL.A.EXECUTION_ERROR")
```

---

### 错误处理决策树（Pipeline 层）

```
AgentError 到达 Pipeline / StageExecutor
│
├─ recovery == A（立即可恢复）
│   ├─ LLM.A.TRANSIENT / LLM.A.RATE_LIMITED
│   │   └─ LLMGateway 内部已退避重试 → 若仍失败切换 provider
│   │       └─ 所有 provider 失败 → 升级为 LLM.C.ALL_PROVIDERS_FAILED
│   ├─ LLM.A.CONTEXT_TOO_LONG
│   │   └─ ContextManager.trim_to_max_tokens() → 重试当前推理轮
│   ├─ LLM.A.RESPONSE_PARSE
│   │     └─ 先本地修复，比如补全JSON，保证完整性→ 重试当前推理轮
│   │   		└─ self-repair（修正 assistant 消息）→ 重试当前推理轮
│   │       	└─ 失败 → 切换 provider
│   ├─ TOOL.A.TIMEOUT
│   │   └─ ToolRegistry 退避重试 → 超出次数升级为 TOOL.C.TIMEOUT_EXHAUSTED
│   ├─ TOOL.A.EXECUTION_ERROR / TOOL.A.ARGUMENT_ERROR
│   │   └─ 错误信息注入上下文（ResultInjected）→ 继续推理循环
│   ├─ SYSTEM.A.MAX_ITERATIONS
│   │   └─ Planner.revise(trigger=STAGE_INFEASIBLE) → TaskPlanRevised → 重新执行该 Stage
│   └─ SYSTEM.A.STAGE_INFEASIBLE
│       └─ Planner.revise(trigger=STAGE_INFEASIBLE) → TaskPlanRevised → 重新执行该 Stage
│
├─ recovery == B（等待后可恢复）
│   └─ 所有 B 类错误
│       └─ TaskPaused(E10) → 等待 UserResumeRequestProvided(E5)
│           └─ TaskResumed → 继续当前步骤
│
└─ recovery == C（不可恢复）
    ├─ TOOL.C.TIMEOUT_EXHAUSTED / TOOL.C.NOT_FOUND / TOOL.C.PERMISSION_DENIED
    │   └─ 注入错误信息 → Planner.revise(trigger=STAGE_INFEASIBLE) → TaskPlanRevised
    │       └─ 若 plan_retries 超限 → TaskTerminated(E13)
    ├─ LLM.C.AUTH_FAILED / LLM.C.CONFIG_ERROR
    │   └─ 跳过 provider → 若全部失败 → LLM.C.ALL_PROVIDERS_FAILED → TaskTerminated(E13)
    ├─ LLM.C.ALL_PROVIDERS_FAILED
    │   └─ TaskTerminated(E13)
    └─ SYSTEM.C.*
        └─ TaskTerminated(E13)
```

# 推理轨迹

## 用户问题重写

| #    | 分析报告字段                    | 是否使用(Y/N)       |
| ---- | ------------------------------- | ------------------- |
| 1    | description                     | Y                   |
| 2    | task_type                       | N ?考虑是否需要使用 |
| 3    | intent                          | Y                   |
| 4    | task_goal                       | Y                   |
| 5    | complexity                      | N                   |
| 6    | required_tools                  | N                   |
| 7    | output_constraints              | Y                   |
| 8    | notes                           | Y                   |
| 9    | related_user_preference_entries | Y                   |
| 10   | related_knowledge_entries       | Y                   |
| 11   | entities                        | N                   |
| 12   | action_constraints              | Y                   |
| 13   | tool_matches                    | N                   |
| 14   | risks                           | Y                   |
|      |                                 |                     |
|      |                                 |                     |
|      |                                 |                     |
|      |                                 |                     |
|      |                                 |                     |

## 执行计划Prompt

