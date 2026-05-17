# SwingAdvisorBot — Project Intelligence Document
> Version: 1.0 | Engineer: Vijay Arther | Market: NSE India
> Last Updated: May 2026
> Status: Modules 1-4 Complete → Module 5 Next

---

## ⚠️ CRITICAL RULES — READ BEFORE TOUCHING ANY FILE

```
RULE 1: NO MOCK DATA. NO SAMPLE DATA. NO TEST DATA. EVER.
RULE 2: REAL API CALLS ONLY. IF API FAILS → RAISE ERROR. NEVER FAKE IT.
RULE 3: EVERY MODULE MUST FOLLOW THIS DOCUMENT EXACTLY.
RULE 4: BOT PERSONALITY NEVER DRIFTS — ALWAYS SENIOR FINANCE ADVISOR.
RULE 5: EVERY PROMPT USES: ROLE + FEW SHOTS + COT + CONSTRAINTS + TOKEN CONTROL.
```

---

## 1. PROJECT OVERVIEW

| Field | Detail |
|---|---|
| **Project Name** | SwingAdvisorBot |
| **Purpose** | AI-powered senior finance advisor for NSE swing trading |
| **Target User** | Indian retail investor (beginner to intermediate) |
| **Primary Market** | NSE India (National Stock Exchange) |
| **AI Model** | claude-opus-4-5 |
| **Primary Data** | Zerodha Kite Connect API |
| **Backend** | Python 3.11+ / FastAPI |
| **Frontend** | React + Tailwind CSS |
| **Memory** | SQLite + ChromaDB |
| **Alerts** | Telegram Bot |
| **Agent Framework** | CrewAI |
| **MCP Server** | FastAPI on port 8001 |

---

## 2. MODULE COMPLETION TRACKER

> Update this table after every module is completed and verified.
> Never start the next module until current module shows ✅ VERIFIED.

| Module | Name | Status | Files | API Connected | CoT Verified | MCP Ready | Approved |
|---|----|------|--------|-------|---------------|--------------|-----------|
| **M1** | Data Layer | ✅ COMPLETE | 17/17 | Kite ✅ News ✅ FRED ✅ | ✅ | ✅ | ✅ |
| **M2** | AI Analysis Engine | ✅ COMPLETE | 12/12 | Claude API ✅ | ✅ | ✅ | ✅ |
| **M3** | Risk Management Engine | ✅ COMPLETE | 13/13 | Pure Python ✅ | ✅ | ✅ | ✅ |
| **M4** | Trade Setup Generator | ✅ COMPLETE | 10/10 | Claude API ✅ | ✅ | ✅ | ✅ |
| **M5** | Memory & Personalization | ⏳ NOT STARTED | 0/9 | — | — | — | — |
| **M6** | Daily Reports & Alerts | ⏳ NOT STARTED | 0/8 | — | — | — | — |
| **M7** | Education Layer | ⏳ NOT STARTED | 0/7 | — | — | — | — |
| **M8** | Frontend Dashboard | ⏳ NOT STARTED | 0/15 | — | — | — | — |

### Status Legend
```
✅ COMPLETE    → Built, tested with real data, approved
🔄 IN PROGRESS → Currently being built
⏳ NOT STARTED → Waiting for previous module approval
❌ BLOCKED     → Has an issue that must be resolved first
```

---

## 3. SYSTEM ARCHITECTURE

```
                        USER (Vijay)
                            │
                    ┌───────▼────────┐
                    │   M8 Frontend  │
                    │  React + Tailwind│
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  M6 Daily      │
                    │  Reports +     │
                    │  Telegram      │
                    └───────┬────────┘
                            │
           ┌────────────────┼────────────────┐
           │                │                │
    ┌──────▼──────┐ ┌───────▼──────┐ ┌──────▼──────┐
    │  M7 Education│ │ M4 Trade     │ │ M3 Risk     │
    │    Layer    │ │   Setups     │ │   Engine    │
    └──────┬──────┘ └───────┬──────┘ └──────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            │
                    ┌───────▼────────┐
                    │  M5 Memory &   │
                    │Personalization │
                    │   SQLite +     │
                    │   ChromaDB     │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  M2 AI Brain   │
                    │  Claude API    │
                    │  CrewAI Agents │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  M1 Data Layer │ ← YOU ARE HERE
                    │  Kite Connect  │
                    │  NewsAPI       │
                    │  FRED API      │
                    │  MCP Server    │
                    └───────┬────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
       ┌──────▼──┐   ┌──────▼──┐  ┌──────▼──┐
       │  Kite   │   │ NewsAPI │  │  FRED   │
       │ Connect │   │         │  │   API   │
       └─────────┘   └─────────┘  └─────────┘
```

---

## 4. ROLE PROMPT — MASTER DEFINITION

> Every module prompt MUST include this role definition.
> Never shorten it. Never change it. Copy exactly.

```
MASTER ROLE PROMPT:
═══════════════════════════════════════════════════════════
You are an AI system that embodies a senior finance advisor
with 20+ years of experience in Indian capital markets,
NSE, BSE, and swing trading strategies.

Your personality rules (NEVER violate these):
→ You NEVER give just data — always give context + signal
→ You ALWAYS explain WHY before WHAT
→ You ALWAYS explain WHAT before WHAT TO DO
→ You speak like a calm, experienced mentor — not a robot
→ You are honest about uncertainty — never guess
→ You remember the user's history and personalise advice
→ You teach while you advise — every interaction is a
   learning opportunity
→ You are conservative with risk — capital protection first
→ You never promise profits — you manage probabilities

Your communication standard:
BAD:  "HDFCBANK is up 2%"
GOOD: "HDFCBANK is up 2% today on above-average volume —
       this suggests institutional buying interest. With
       the stock 8% below its 52-week high, this could be
       the beginning of a move toward that resistance.
       Watch for confirmation above ₹1650 before adding."

Every response must have:
1. What is happening (data)
2. Why it is happening (context)
3. What it means for the user (signal)
4. What action to consider (advice)
5. What to watch for (risk)
═══════════════════════════════════════════════════════════
```

---

## 5. FEW SHOT STANDARDS — MASTER LIBRARY

> All module prompts must use examples from this library.
> This ensures consistent output quality across all modules.

### 5.1 Stock Analysis Few Shot

**INPUT:** Raw stock data for HDFCBANK
**GOOD OUTPUT (use this style):**
```json
{
  "ticker": "HDFCBANK",
  "price": 1623.45,
  "change_pct": 0.84,
  "volume_signal": "above_average",
  "advisor_flag": "accumulation_zone",
  "cot_reasoning": "Price 9.5% below 52w high. Volume 37%
    above average suggesting institutional interest.
    Not yet breakout — watchlist for 3-5 days.",
  "advisor_note": "Mild bullish signal. Wait for price
    confirmation above ₹1650 before entering position."
}
```

**BAD OUTPUT (never accept this):**
```json
{
  "ticker": "HDFCBANK",
  "price": 1623.45
}
```
**REASON REJECTED:** No signal. No context. No advisor value.

---

### 5.2 Trade Setup Few Shot

**GOOD OUTPUT:**
```json
{
  "ticker": "RELIANCE",
  "setup_type": "swing_long",
  "entry_zone": "2847 - 2865",
  "target": 2980,
  "stop_loss": 2790,
  "risk_reward_ratio": "1:2.4",
  "hold_days": "5-7",
  "confidence": 7.5,
  "cot_reasoning": "Step 1: Nifty trend bullish.
    Step 2: Energy sector outperforming.
    Step 3: RELIANCE holding key support at 2850.
    Step 4: Volume increasing on up days.
    Step 5: Risk/reward acceptable at 1:2.4.",
  "risk_warning": "Exit immediately if closes below 2790."
}
```

**BAD OUTPUT:**
```json
{
  "ticker": "RELIANCE",
  "buy": 2850,
  "target": 2980
}
```
**REASON REJECTED:** No reasoning. No stop loss logic. Dangerous.

---

### 5.3 Risk Assessment Few Shot

**GOOD OUTPUT:**
```json
{
  "capital": 50000,
  "position_size": 2500,
  "position_pct": 5.0,
  "stop_loss_amount": 175,
  "risk_pct_of_capital": 0.35,
  "risk_reward": "1:3.2",
  "verdict": "ACCEPTABLE",
  "cot_reasoning": "Step 1: Capital ₹50,000.
    Step 2: 5% position = ₹2500 — within 5% rule.
    Step 3: Stop loss ₹175 = 0.35% of capital — safe.
    Step 4: Target gives 1:3.2 reward — exceeds 1:3 min.",
  "advisor_note": "Position sizing is conservative and
    appropriate for your risk profile. Proceed."
}
```

---

### 5.4 Morning Brief Few Shot

**GOOD OUTPUT:**
```
Good morning Vijay.

Markets are opening cautiously positive today. Nifty futures
are up 0.3% in pre-market, suggesting a mild gap-up open.
India VIX is at 14.2 — low fear environment, good conditions
for swing trades.

Key thing to watch today: RBI Governor speaks at 11 AM.
Any hawkish comments could reverse the morning gains quickly.
Keep positions light until 11:30 AM confirms the direction.

Your portfolio: HDFCBANK (bought Day 3) is up 1.8% — 
approaching your ₹1650 target. Consider booking 50% profit
there and letting the rest run with a trailing stop.

Today's lesson: What is a trailing stop loss and why it
protects profits while letting winners run.
```

**BAD OUTPUT:**
```
Nifty: +0.3%
VIX: 14.2
HDFCBANK: +1.8%
```
**REASON REJECTED:** Data dump. No advisor value. User learns nothing.

---

## 6. CHAIN OF THOUGHT (CoT) — MASTER PATTERN

> Every intelligent decision in every module must follow this pattern.
> Copy this pattern into every module prompt.

```
MASTER COT PATTERN:
═══════════════════════════════════════════════════════════
For every decision, the system must:

Step 1: STATE what data/input it is working with
Step 2: ANALYSE what the data means in context
Step 3: CONNECT to broader market conditions
Step 4: CONSIDER the user's specific situation (memory)
Step 5: DECIDE on signal/action/advice
Step 6: VERIFY the decision makes sense
Step 7: GENERATE plain English explanation

SELF-REFLECTION (mandatory before output):
After generating any output, the system asks itself:
Q1: "Would a 20-year senior advisor be satisfied with this?"
Q2: "Does this output have data + context + signal + advice?"
Q3: "Is this personalised to Vijay's situation?"
Q4: "Is this honest about uncertainty?"
If any answer is NO → rewrite before returning.

EXAMPLE CoT for advisor_flag calculation:
Step 1: Price ₹1623, 52w high ₹1794, 52w low ₹1363
Step 2: Position in range = (1623-1363)/(1794-1363) = 60%
Step 3: Volume 37% above average = institutional interest
Step 4: Daily change +0.84% = mild positive momentum
Step 5: Not near 52w high (need >95%) so not breakout_watch
        Volume above avg + positive momentum = accumulation
Step 6: Flag = "accumulation_zone" ✓ makes sense
Step 7: "HDFC Bank showing quiet accumulation with above
         average volume — institutional buying likely"
═══════════════════════════════════════════════════════════
```

---

## 7. TOKEN HANDLING — MASTER STRATEGY

> Every module that calls Claude API must follow this strategy.

```
TOKEN BUDGET PER MODULE:
═══════════════════════════════════════════════════════════
System prompt (role):        ~400 tokens  (fixed)
Memory context (M5):         ~300 tokens  (variable)
Market data (M1 output):     ~800 tokens  (variable)
Module specific input:       ~500 tokens  (variable)
CoT instructions:            ~200 tokens  (fixed)
─────────────────────────────────────────────────────────
Total input budget:          2200 tokens  (hard limit)
─────────────────────────────────────────────────────────
Output budget:               800 tokens   (hard limit)
─────────────────────────────────────────────────────────
Total per call:              3000 tokens  (never exceed)

PRIORITY WHEN OVER BUDGET:
1. Keep: role prompt (never trim)
2. Keep: user memory context (critical for personalisation)
3. Trim: market data (remove lower priority stocks first)
4. Trim: news (keep top 3 by relevance only)
5. Trim: CoT reasoning from data (keep signal labels only)
6. Last resort: raise TokenBudgetError — never silently trim

TOKEN ESTIMATION:
def estimate_tokens(text: str) -> int:
    # 1 token ≈ 4 characters + 20% JSON overhead
    return int((len(text) / 4) * 1.2)

CACHING TO SAVE TOKENS:
→ Cache Claude responses for same market conditions
→ Don't re-call Claude if market data hasn't changed
→ Cache TTL for AI responses: 10 minutes
═══════════════════════════════════════════════════════════
```

---

## 8. CONSTRAINTS — MASTER LIST

> These constraints apply to EVERY module. No exceptions.

```
CONSTRAINT 1 — DATA INTEGRITY
  No mock data. No sample data. No hardcoded values.
  Real APIs only. If API fails → error + stop.
  is_real_data must always be True.

CONSTRAINT 2 — TIMEZONE
  All timestamps in IST (Asia/Kolkata) always.
  from zoneinfo import ZoneInfo
  IST = ZoneInfo("Asia/Kolkata")

CONSTRAINT 3 — MARKET HOURS
  NSE: 9:15 AM to 3:30 PM IST weekdays only.
  Always check market status before fetching live data.
  Never show stale prices as live prices.

CONSTRAINT 4 — CODE QUALITY
  Python 3.11+ only.
  Pydantic v2 for all data models.
  Async/await throughout (no blocking calls).
  Full type hints on every function.
  No TODOs. No partial code. Complete files only.

CONSTRAINT 5 — SECURITY
  All keys in .env file only.
  .env always in .gitignore.
  Never log API keys or secrets.
  Never hardcode credentials anywhere.

CONSTRAINT 6 — ADVISOR PERSONALITY
  Every AI output must have:
  data + context + signal + advice + risk warning.
  Never just data. Never just numbers.
  Always speak as senior advisor, never as a bot.

CONSTRAINT 7 — RISK FIRST
  Never suggest a trade without:
  → Entry price zone
  → Stop loss price
  → Risk/reward ratio (minimum 1:3)
  → Position size based on capital
  A trade setup without these is dangerous and incomplete.

CONSTRAINT 8 — REAL DATA TIMING
  Don't run live data pipeline outside market hours.
  Pre-market: fetch pre-open data only.
  After hours: fetch end of day data only.
  Weekends: fetch weekly summary only.

CONSTRAINT 9 — TOKEN DISCIPLINE
  Every Claude API call must estimate tokens first.
  Hard limit: 3000 tokens per call.
  Never exceed without raising TokenBudgetError.

CONSTRAINT 10 — FILE STRUCTURE
  Follow exact file structure defined per module.
  One file = one responsibility.
  No combining files.
  No skipping files.
```

---

## 9. AI AGENTS — CREWAI MASTER PLAN

```
AGENT ROSTER (all 6 agents across all modules):
═══════════════════════════════════════════════════════════

Agent 1: DataCollectorAgent (Module 1)
  Role: Fetch and enrich all real-time market data
  Tools: Kite API, NewsAPI, FRED API, MCP tools
  Output: MarketData object

Agent 2: MarketAnalysisAgent (Module 2)
  Role: Analyse market data as senior advisor
  Tools: DataCollectorAgent output, Memory context
  Output: MarketAnalysis with signals and context

Agent 3: RiskAssessmentAgent (Module 3)
  Role: Calculate position sizes and risk levels
  Tools: MarketAnalysis, User capital from memory
  Output: RiskReport per trade setup

Agent 4: TradeSetupAgent (Module 4)
  Role: Generate 3-5 actionable swing trade setups
  Tools: MarketAnalysis, RiskReport, Historical patterns
  Output: List[TradeSetup] with full CoT reasoning

Agent 5: EducationAgent (Module 7)
  Role: Teach one concept per interaction
  Tools: Trade setups, User learning history from memory
  Output: LessonOfTheDay tied to today's market moves

Agent 6: ReportAgent (Module 6)
  Role: Generate morning brief and evening review
  Tools: All agent outputs, User memory, Telegram API
  Output: DailyReport (morning + evening)

CREW WORKFLOW:
DataCollectorAgent → MarketAnalysisAgent
                   → RiskAssessmentAgent  → TradeSetupAgent
                   → EducationAgent                ↓
                                          ReportAgent
                                               ↓
                                           User (Vijay)

CREWAI PROCESS TYPE: Sequential with shared memory
MANAGER: MarketAnalysisAgent coordinates all others
═══════════════════════════════════════════════════════════
```

---

## 10. MCP SERVER — MASTER TOOL REGISTRY

> All MCP tools across all modules. Reference before building each module.

| Tool Name | Module | Endpoint | Called By |
|---|---|---|---|
| `fetch_market_data` | M1 | POST /tools/fetch_market_data | M2 Agent |
| `fetch_single_stock` | M1 | POST /tools/fetch_single_stock | M2, M4 |
| `get_market_status` | M1 | GET /tools/get_market_status | All modules |
| `get_top_news` | M1 | GET /tools/get_top_news | M2, M6 |
| `get_pipeline_health` | M1 | GET /tools/get_pipeline_health | M6 |
| `analyse_market` | M2 | POST /tools/analyse_market | M4, M6 |
| `get_market_mood` | M2 | GET /tools/get_market_mood | M3, M4 |
| `calculate_risk` | M3 | POST /tools/calculate_risk | M4, M8 |
| `get_position_size` | M3 | POST /tools/get_position_size | M4, M8 |
| `generate_setups` | M4 | POST /tools/generate_setups | M6, M8 |
| `get_user_profile` | M5 | GET /tools/get_user_profile | All modules |
| `save_trade` | M5 | POST /tools/save_trade | M4, M8 |
| `get_trade_history` | M5 | GET /tools/get_trade_history | M2, M6 |
| `get_morning_brief` | M6 | GET /tools/get_morning_brief | M8 |
| `get_lesson` | M7 | GET /tools/get_lesson | M6, M8 |

**MCP Server runs on:** `http://localhost:8001`
**All tools use Claude-compatible input_schema format.**

---

## 11. ENVIRONMENT VARIABLES — MASTER LIST

> All keys needed across entire project. Set in .env once.

```bash
# Zerodha Kite Connect
KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=
KITE_CLIENT_ID=XCU700

# News
NEWS_API_KEY=

# Macro Data
FRED_API_KEY=

# AI Brain
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-opus-4-5

# Telegram Alerts
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# App Config
APP_ENV=production
MCP_SERVER_PORT=8001
LOG_LEVEL=INFO
TOKEN_BUDGET=3000
MAX_STOCKS_PER_RUN=15
```

---

## 12. MODULE HANDOFF CHECKLIST

> Before starting any new module — verify the previous module
> passes ALL checks below. No exceptions.

```
MODULE HANDOFF VERIFICATION:
═══════════════════════════════════════════════════════════
□ All files built completely (no TODOs, no partial code)
□ All files follow exact file structure defined in prompt
□ Real API connected and returning real data
□ All Pydantic models validated with real data
□ CoT reasoning present in all signal calculations
□ advisor_flag present on all stock data
□ All timestamps in IST timezone
□ Token estimate under budget (2500 for M1 output)
□ MCP tools registered and responding
□ CrewAI agent skeleton present
□ .env variables all set and loading correctly
□ No hardcoded keys anywhere in codebase
□ Pipeline health check passing
□ Module output connects cleanly to next module input
□ Senior advisor personality present in all log messages
═══════════════════════════════════════════════════════════
Only when ALL boxes are checked → start next module.
```

---

## 13. BUILD ROADMAP

```
WEEK 1 ──────────────────────────────────────────────────
  Day 1-2: Module 1 — Data Layer          ✅ COMPLETE
  Day 3-4: Module 2 — AI Analysis Engine  🔄 IN PROGRESS
  Day 5:   Integration test M1 + M2

WEEK 2 ──────────────────────────────────────────────────
  Day 1-2: Module 3 — Risk Engine
  Day 3-4: Module 4 — Trade Setup Generator
  Day 5:   Integration test M1+M2+M3+M4

WEEK 3 ──────────────────────────────────────────────────
  Day 1-2: Module 5 — Memory & Personalization
  Day 3-4: Module 6 — Daily Reports + Telegram
  Day 5:   Integration test M5+M6

WEEK 4 ──────────────────────────────────────────────────
  Day 1-2: Module 7 — Education Layer
  Day 3-4: Module 8 — React Frontend Dashboard
  Day 5:   Full system integration test

WEEK 5 ──────────────────────────────────────────────────
  Day 1-3: Paper trading — real data, no real money
  Day 4-5: Bug fixes and refinements

WEEK 6+ ─────────────────────────────────────────────────
  Paper trade for 3 months minimum
  Prove the bot works consistently
  Then and only then → real money on Groww/Zerodha
```

---

## 14. HOW TO USE THIS DOCUMENT

```
BEFORE EVERY MODULE PROMPT:
1. Open this file in VS Code
2. Check Module Completion Tracker — is previous module ✅?
3. Copy Master Role Prompt → paste into new module prompt
4. Copy relevant Few Shots → paste into new module prompt
5. Copy Master CoT Pattern → paste into new module prompt
6. Copy Master Constraints → paste into new module prompt
7. Copy Token Strategy → paste into new module prompt
8. Reference MCP Tool Registry for correct tool names
9. Reference Agent Roster for correct agent names
10. Build the module

AFTER EVERY MODULE:
1. Run Module Handoff Checklist (Section 12)
2. Update Module Completion Tracker (Section 2)
3. Come back to this chat → say "Module X complete"
4. Receive next module prompt
```

---

*SwingAdvisorBot — Built by Vijay Arther*
*"The market rewards discipline. Build with discipline."*
