# Earned Limits for Instinct (concept demo)

**Live demo:** https://mohith26.github.io/instinct-earned-limits/

Unofficial concept demo by Mohith Gajjela. Not affiliated with Instinct. All data is synthetic.

## What it is

Instinct's Link balance (Oct 26, 2026) lets you set one per-purchase limit: anything under it is bought without asking, anything over it waits for your approval. One flat number has to cover a $4 coffee reorder and a $96 pair of shoes Instinct suggested from a store you've never used.

**Earned limits** replaces the single number with limits each user earns from their own approve / decline history, with a statistical guarantee:

> At most 1 in 20 of your purchase requests end in a no-ask buy you wouldn't have approved, with 90% confidence.

It lives on a new **Spending** page in the Instinct web app, next to Workspace and Vault, built on Instinct's own design tokens (colors, radii, spacing, row and section components copied from the production bundle at app.instinct.com).

## How it works

1. **Risk score.** A L2-regularized logistic regression (Newton/IRLS, `core/earned.js`) estimates the probability you'd decline a purchase request, from: amount relative to your usual spend in that category, who initiated it (you, a routine, or an Instinct suggestion), same item as before, new store, price change since last time, time-sensitivity, group chat, and your own shrunk decline rates. Personal features are frozen from a warm-up window before any calibration label is used.
2. **Certification (Learn then Test).** The rule is: buy without asking iff amount <= your hard limit, amount <= 2.5x your usual in that category, the category isn't on always-ask (travel) or still thin (< 8 requests), and score <= tau. For tau on an increasing grid, test H0: R(tau) > alpha, where R is the per-request rate of "bought without asking AND you'd have said no", with an exact binomial p-value on your own past answers. Fixed-sequence testing stops at the first tau that fails, which controls family-wise error, so P(R(tau_hat) > alpha) <= delta (Angelopoulos, Bates, Candes, Jordan, Lei 2021, "Learn then Test").
3. **Per-category limits** shown in the UI are derived from tau_hat: the largest amount at which a usual purchase in that category still scores under tau, plus which unusual signals (I suggested it, new store, price up 20%+) would make Instinct ask anyway.
4. **Trust check (anytime-valid).** After certification, every answer and every return feeds a betting e-process testing whether your declines run above what the model expects (residual mean above its calibration level). Under no change it trips with probability <= 5% no matter how often it is checked (Ville's inequality); when it trips, earned limits pause and Instinct goes back to asking.
5. **Replay.** The last 14 days (after Link balance launched, not used for certifying) are replayed under the flat $150 limit and under earned limits, with every differing decision listed.

## Results (synthetic, `data/validation.json`)

Validated on 300 fresh simulated users (model trained on a separate 400):

| | value |
|---|---|
| Model AUC on a fresh cohort | 0.84 |
| Users who earned any limit after 35 days | 68% |
| Earned users whose TRUE unwanted-no-ask rate <= 1 in 20 | 99.5% (guarantee: >= 90%) |
| Users for whom a flat $150 limit exceeds 1 in 20 | 93% |
| Median share of approval texts skipped, flat dollar limit certified the same way | 13% |
| Median share of approval texts skipped, earned limits (all users) | 31% |
| Trust check false pauses over 120 days, no change | 1% |
| Trust check pauses after a simulated change in habits | 91.5%, median 29 days |

Demo account (seeded): over the 14 replay days, the flat $150 limit bought 58 of 74 requests without asking and 8 were unwanted ($465.73); earned limits would have bought 56 with 3 unwanted ($101.73).

## Run it

```
node scripts/generate.mjs   # seeded data, model training, validation -> data/
node --test test/            # 12 unit tests (binomial math, IRLS recovery, LTT fixed sequence, guarantee check over simulated users, e-process false-alarm rate, ...)
```
Then open `index.html` (static, CDN fonts and icons only; `#preview`, `#sim`, `#diff` deep links open the approval text, the habit-change simulation and the decision diff).

## Honest limitations

- All users, purchases and answers are simulated; the generative model (`core/earned.js`, `simulateProposals`) is mine, not Instinct's. Real decline behavior is burstier and less stationary.
- The guarantee is per user and assumes future requests are exchangeable with the calibration window. The trust check is what covers drift, and it is slower for small shifts (median 29 days in the simulation).
- Personal guarantees need data: an account needs roughly 45+ answered purchase requests before anything is earned. Pooling across similar users would earn limits sooner but would weaken the guarantee to a population-level one.
- "Unwanted" for a no-ask buy is approximated by returns and complaints, which under-count quiet regret.
- Fonts are open substitutes (Outfit for KMR Melange Grotesk, Newsreader for Aime); icons are Lucide at stroke 1.5.

## Novelty note

As of Nov 10, 2026, Instinct's Link balance offers a single static per-purchase limit (announced Oct 26 by Noah Shinn and @instinct; Link's setup guide), and vault approvals offer Allow once / Always allow / Don't allow. I found no Instinct announcement of learned or per-category limits. Meta's Muse says it "learns over time which decisions need your sign off" without describing a method or guarantee; Stripe has said budget controls for agents are coming. Using conformal / Learn-then-Test risk control to decide when an agent must ask a human is known in research (e.g. KnowNo, Ren et al. 2023, for robots asking for help); applying it per user to agent spending with an anytime-valid pause is, as far as I found, new.
