# OpenAI_Models

_Sidecar: mechanically extracted text from `OpenAI_Models.pdf`. Original PDF preserved._

OpenAI Model Cheatsheet
Only Plus subscribers get access to the GPT-4, GPT-4o and GPT-4.5 slots (and DALL·E 3).

Model          Limit           Reset            Description
GPT-4          40 msgs         every 3 h        Advanced general-purpose with strong CoT reasoning; will retire from
                                                ChatGPT on Apr 30, 2025.
GPT-4o         80 msgs         every 3 h        “Omni” successor to GPT-4 with native vision/audio support, faster &
                                                more consistent.
GPT-4.5        ~50 msgs        per week         Feb 2025 research preview; excels at conversation & EQ, limited
(Orion)                                         chain-of-thought.
o4-mini        150 msgs        daily at 00:00   Lightweight GPT-4o variant for general use; lower latency, daily cap
                               UTC              instead of rolling window.
o4-mini-high   50 msgs         daily at 00:00   High-performance o4-mini tuned for coding, STEM & logical tasks.
                               UTC
o3             50 msgs         per week         Original o-series model; legacy; weaker than o4 variants.
o3-mini        150 msgs        daily at 00:00   Compact model for fast answers & summaries; less depth on complex CoT
                               UTC              reasoning.
o3-mini-high   50 msgs         daily at 00:00   Enhanced o3-mini variant optimized for coding/debugging tasks.
                               UTC
o1-preview     50 msgs         per week         Very early public reasoning model; largely superseded by o3 series.
DALL·E 3       40 prompts      every 3 h; 200   In-chat image generation; supports free-form and in-context ops.
                               imgs/day
Deep           10 queries      per month        Academic-style search & analysis tool.
Research
Projects       20 projects ×   N/A              File/data sandbox: 512 MB/file, 2M tokens total.
               20 files
