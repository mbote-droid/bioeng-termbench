# Diagrams

Mermaid sources; GitHub renders them in place.

## Components

```mermaid
flowchart LR
  subgraph agent["agent container"]
    I[instruction.md + example inputs] --> S[submission]
  end
  subgraph verifier["verifier container"]
    T[test.sh] --> CLI[bioeng grade]
    CLI --> REG[registry]
    REG --> TP[TaskPlugin tXX]
    TP --> M[models: ECG, PK, glucose, bioheat]
    TP --> C[checks library]
    TP --> R[Runner]
    R -->|uid sandbox, rlimits| SUB[submission process]
    CLI --> SC[scoring: Verdict]
    SC --> RW[(reward.txt / reward.json / verdict.json)]
  end
  S -->|artifact| SUB
  SEEDS[(seeds 0700)] --> CLI
```

## Grading lifecycle (one case)

```mermaid
sequenceDiagram
  participant G as grader (root)
  participant T as TaskPlugin
  participant R as Runner
  participant S as submission (sandbox)
  G->>T: make_case(seed)
  T-->>G: case (truth, in memory)
  G->>T: write_inputs(case, input_dir)
  loop each RunPlan
    G->>R: run(spec)
    R->>S: start (new session, rlimits, scrubbed env)
    S-->>R: exit / timeout
    R->>R: kill process group, reap sandbox uid
    R-->>G: RunResult
    G->>G: reject symlinked outputs
  end
  G->>T: checks().evaluate(ctx)
  T-->>G: CheckResults (gates skip dependants)
```

## t03 software in the loop

```mermaid
sequenceDiagram
  participant P as patient model (grader)
  participant C as CGM model (grader)
  participant K as controller (submission)
  loop every 5 min
    P->>C: interstitial glucose
    C->>K: {"t_min", "cgm_mg_dL" or null, "basal", "pump_max", "exercise_announced"}
    K-->>P: {"insulin_mU_per_min"}  (deadline 2 s)
    P->>P: RK4, 10 x 0.5 min, rate held constant
  end
  Note over P: records written after the controller is killed,<br/>into a directory the sandbox user cannot write
```

## t04 verification of the agent's solver

```mermaid
flowchart TB
  subgraph space["space refinement (dt = 1 s)"]
    s1[N=200] --> s2[N=400] --> s3[N=800]
  end
  subgraph time["time refinement (N = 400)"]
    t1[dt=2] --> t2[dt=1] --> t3[dt=0.5]
  end
  s1 & s2 & s3 --> OS[energy order in space >= 1.6]
  t1 & t2 & t3 --> OT[energy order in time >= 1.6]
  s2 --> ACC[accuracy vs Richardson reference]
  s2 --> EB[energy budget closes]
  PLAN[plan.json] --> REF[grader reference at highest and lowest perfusion]
  REF --> COV[coverage] & SAFE[safety] & EFF[efficiency] & CONS[consistency]
```

## Validation campaign (Snakemake)

```mermaid
flowchart LR
  P[validation/probes.py] --> J1[probe jobs x 47]
  G[task grader modules] --> J1
  G --> J2[mutation jobs x 5]
  RT[validation/redteam/*] --> J3[red team]
  J1 --> RES[(results/validation/*.json)]
  J2 --> RES
  J3 --> RES
  RES --> REP[report.py]
  F[FINDINGS.md] --> REP
  REP --> VR[docs/VALIDATION_REPORT.md]
  REP --> MAN[MANIFEST.sha256]
```

## Task plugin class structure

```mermaid
classDiagram
  class TaskPlugin {
    <<abstract>>
    task_id
    version
    timeout_s
    make_case(seed)
    write_inputs(case, dir)
    argv(lang)
    checks() CheckGroup
    languages()
    run_plans(case)
    execute(...)
    grade(submission, seeds, runner)
  }
  class Runner {
    <<protocol>>
    run(spec) RunResult
    start(spec, step_timeout) Session
  }
  TaskPlugin <|-- EcgTask
  TaskPlugin <|-- PkTask
  TaskPlugin <|-- GlucoseTask
  TaskPlugin <|-- AblationTask
  TaskPlugin <|-- PkDebugTask
  Runner <|.. LocalRunner
  Runner <|.. DockerRunner
  TaskPlugin --> Runner
  TaskPlugin --> CheckGroup
  CheckGroup o-- FunctionCheck
```
