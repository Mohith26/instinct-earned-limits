// Seeded data generation + model training + offline validation.
// node scripts/generate.mjs   -> writes data/model.json, data/demo.json, data/validation.json
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import {
  mulberry32, drawUser, simulateProposals, personalContext, featurize, fitLogistic, score,
  populationPrior, certify, decideAuto, conditionalBound, eProcess, residualNull, CATEGORIES, CAT_INDEX, FEATURE_NAMES, trueRisk,
} from '../core/earned.js';

export const TIMELINE = { start: '2026-08-24', warmupEnd: 28, calibEnd: 63, replayEnd: 77 };
const FLAT_GRID = Array.from({ length: 59 }, (_, i) => 10 + i * 5);
const ALPHA = 0.05, DELTA = 0.1, HARD_CAP = 300, STATIC_LIMIT = 150;
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
const out = (name, obj) => writeFileSync(new URL('../data/' + name, import.meta.url), JSON.stringify(obj));

// ---------- 1. training cohort ----------
const rngTrain = mulberry32(1101);
const trainUsers = Array.from({ length: 400 }, () => drawUser(rngTrain));
const trainProps = trainUsers.map((u, i) => simulateProposals(rngTrain, u, 0, TIMELINE.replayEnd, `t${i}_`));
const prior = populationPrior(trainProps.flat().filter((p) => p.day < TIMELINE.warmupEnd));
const X = [], y = [];
trainProps.forEach((props) => {
  const ctx = personalContext(props.filter((p) => p.day < TIMELINE.warmupEnd), prior);
  props.filter((p) => p.day >= TIMELINE.warmupEnd).forEach((p) => { X.push(featurize(p, ctx)); y.push(p.unwanted ? 1 : 0); });
});
const w = fitLogistic(X, y, 1.0);
const means = FEATURE_NAMES.map((_, j) => X.reduce((s, x) => s + x[j], 0) / X.length);
// training log loss + AUC on a held-out cohort (computed below)
out('model.json', { weights: w, features: FEATURE_NAMES, means, prior, trainRows: X.length, trainUsers: trainUsers.length, l2: 1.0 });

// ---------- 2. demo user (the account shown in the UI) ----------
const rngDemo = mulberry32(20261109);
const demoUser = drawUser(rngDemo, {
  activity: 2.6, priceSens: 1.4, suggestAversion: 1.9, newMerchantAversion: 1.0, jumpSens: 4.5,
  catOffset: [-0.7, -0.9, 0.1, -0.6, 0.2, 0.3, 0.5, 1.1],
  catMedian: [124, 29, 42, 22, 140, 380, 74, 96],
});
const demoProps = simulateProposals(rngDemo, demoUser, 0, TIMELINE.replayEnd, 'd');
out('demo.json', { user: demoUser, timeline: TIMELINE, proposals: demoProps, settings: { alpha: ALPHA, delta: DELTA, hardCap: HARD_CAP, staticLimit: STATIC_LIMIT } });

// ---------- 3. validation on a fresh cohort ----------
const rngVal = mulberry32(7);
const testUsers = Array.from({ length: 300 }, () => drawUser(rngVal));
let auc = { pos: [], neg: [] }, logloss = 0, nll = 0;
const perUser = [];
testUsers.forEach((u, i) => {
  const props = simulateProposals(rngVal, u, 0, TIMELINE.calibEnd, `v${i}_`);
  const ctx = personalContext(props.filter((p) => p.day < TIMELINE.warmupEnd), prior);
  const calib = props.filter((p) => p.day >= TIMELINE.warmupEnd).map((p) => ({ ...p, s: score(w, featurize(p, ctx)), usual: ctx.median[CAT_INDEX[p.cat]] }));
  calib.forEach((r) => { (r.unwanted ? auc.pos : auc.neg).push(r.s); logloss += -(r.unwanted ? Math.log(r.s) : Math.log(1 - r.s)); nll++; });
  const cert = certify(calib, { alpha: ALPHA, delta: DELTA, hardCap: HARD_CAP });
  // True risk of the certified rule and of a static $150 limit, by Monte Carlo over
  // the user's own generative process with exact truth probabilities.
  const fresh = simulateProposals(mulberry32(90000 + i), u, TIMELINE.calibEnd, TIMELINE.calibEnd + 400, `f${i}_`);
  let rE = 0, aE = 0, rS = 0, aS = 0;
  fresh.forEach((p) => {
    const s = score(w, featurize(p, ctx));
    if (decideAuto({ ...p, s, usual: ctx.median[CAT_INDEX[p.cat]] }, cert)) { aE++; rE += p.pTrue; }
    if (p.amount <= STATIC_LIMIT) { aS++; rS += p.pTrue; }
  });
  // Baseline A: a flat dollar limit certified the same way (LTT on amount only).
  const flatRows = calib.map((r) => ({ ...r, s: r.amount / HARD_CAP }));
  const flat = certify(flatRows, { alpha: ALPHA, delta: DELTA, hardCap: HARD_CAP, grid: FLAT_GRID.map((L) => L / HARD_CAP), alwaysAsk: [], minPerCat: 0 });
  // Baseline B (oracle, uses truth): best flat limit whose TRUE risk <= alpha.
  let aF = 0, oracleAuto = 0;
  const flatL = flat.tauHat == null ? null : flat.tauHat * HARD_CAP;
  fresh.forEach((p) => { if (flatL != null && p.amount <= flatL) aF++; });
  for (const L of FLAT_GRID) {
    let rr = 0, aa = 0; fresh.forEach((p) => { if (p.amount <= L) { aa++; rr += p.pTrue; } });
    if (rr / fresh.length <= ALPHA) oracleAuto = Math.max(oracleAuto, aa / fresh.length);
  }
  perUser.push({ autoFlat: aF / fresh.length, autoOracleFlat: oracleAuto, nCalib: calib.length, tauHat: cert.tauHat, riskEarned: rE / fresh.length, autoEarned: aE / fresh.length, riskStatic: rS / fresh.length, autoStatic: aS / fresh.length });
});
// AUC by rank
const all = auc.pos.map((s) => [s, 1]).concat(auc.neg.map((s) => [s, 0])).sort((a, b) => a[0] - b[0]);
let rankSum = 0; all.forEach((v, idx) => { if (v[1]) rankSum += idx + 1; });
const AUC = (rankSum - auc.pos.length * (auc.pos.length + 1) / 2) / (auc.pos.length * auc.neg.length);
const earned = perUser.filter((r) => r.tauHat != null);
const med = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : null; };
const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;

// ---------- 4. monitor: false alarms without drift, detection delay with drift ----------
function monitorRun(seed, withDrift) {
  const rng = mulberry32(seed);
  const u = drawUser(rng, { activity: 2.0 });
  const props = simulateProposals(rng, u, 0, TIMELINE.calibEnd, 'm');
  const ctx = personalContext(props.filter((p) => p.day < TIMELINE.warmupEnd), prior);
  const calib = props.filter((p) => p.day >= TIMELINE.warmupEnd).map((p) => ({ ...p, s: score(w, featurize(p, ctx)), usual: ctx.median[CAT_INDEX[p.cat]] }));
  const cert = certify(calib, { alpha: ALPHA, delta: DELTA, hardCap: HARD_CAP });
  if (cert.tauHat == null) return null;
  const nul = residualNull(calib), q0 = nul.m0;
  if (withDrift) u.drift = { day: TIMELINE.calibEnd + 10, shift: 1.2, cats: ['food', 'clothing', 'gifts'], catShift: 1.0 };
  const future = simulateProposals(rng, u, TIMELINE.calibEnd, TIMELINE.calibEnd + 120, 'mf');
  const labels = [], days = [];
  future.forEach((p) => {
    const s = score(w, featurize(p, ctx));
    // every request yields a label: asked ones get approve/deny/ignore; auto-bought
    // ones get an outcome (kept, or returned / flagged as unwanted)
    labels.push((p.unwanted ? 1 : 0) - s); days.push(p.day);
  });
  const ep = eProcess(labels, q0, 0.05, -1);
  const pre = labels.filter((_, k) => days[k] < TIMELINE.calibEnd + 10), post = labels.filter((_, k) => days[k] >= TIMELINE.calibEnd + 10);
  const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  return { q0, tripped: ep.tripped != null, delayDays: ep.tripped != null ? days[ep.tripped] - (TIMELINE.calibEnd + 10) : null, checks: labels.length,
    preRate: avg(pre), postRate: avg(post) };
}
const noDrift = [], drift = [];
for (let s = 0; s < 200; s++) { const a = monitorRun(5000 + s, false); if (a) noDrift.push(a); const b = monitorRun(5000 + s, true); if (b) drift.push(b); }
const delays = drift.filter((d) => d.tripped && d.delayDays >= 0).map((d) => d.delayDays);

const validation = {
  alpha: ALPHA, delta: DELTA, hardCap: HARD_CAP, staticLimit: STATIC_LIMIT,
  users: perUser.length,
  model: { auc: AUC, logLoss: logloss / nll, calibRows: nll },
  earnedShare: earned.length / perUser.length,
  coverage: earned.length ? earned.filter((r) => r.riskEarned <= ALPHA).length / earned.length : null,
  coverageAll: perUser.filter((r) => (r.tauHat == null ? 0 : r.riskEarned) <= ALPHA).length / perUser.length,
  medianRiskEarned: med(earned.map((r) => r.riskEarned)),
  medianAutoEarned: med(earned.map((r) => r.autoEarned)),
  medianRiskStatic: med(perUser.map((r) => r.riskStatic)),
  medianAutoStatic: med(perUser.map((r) => r.autoStatic)),
  medianAutoFlatCertified: med(perUser.map((r) => r.autoFlat)),
  medianAutoOracleFlat: med(perUser.map((r) => r.autoOracleFlat)),
  medianAutoEarnedAll: med(perUser.map((r) => (r.tauHat == null ? 0 : r.autoEarned))),
  staticOverAlpha: perUser.filter((r) => r.riskStatic > ALPHA).length / perUser.length,
  meanCalibRowsEarned: earned.length ? mean(earned.map((r) => r.nCalib)) : null,
  minCalibRowsEarned: earned.length ? Math.min(...earned.map((r) => r.nCalib)) : null,
  monitor: {
    runsNoDrift: noDrift.length, falseAlarmRate: noDrift.filter((r) => r.tripped).length / noDrift.length,
    runsDrift: drift.length, detectRate: drift.filter((r) => r.tripped).length / drift.length,
    detectWithin30: drift.filter((r) => r.tripped && r.delayDays <= 30).length / drift.length,
    medianDelayDays: med(delays), eps: 0.05, medianNullResidual: med(drift.map((r) => r.q0)),
    medianResidualBefore: med(drift.map((r) => r.preRate)), medianResidualAfter: med(drift.map((r) => r.postRate)),
    drift: 'from day 10: +1.2 logit on every category, +1.0 more on food, clothing, gifts (e.g. user starts a budget)',
  },
};
out('validation.json', validation);
console.log(JSON.stringify(validation, null, 1));

// Bundle for the static page (works from file:// and GitHub Pages without fetch).
{
  const rd = (n) => JSON.parse(readFileSync(new URL('../data/' + n, import.meta.url)));
  writeFileSync(new URL('../data/data.js', import.meta.url),
    'window.EARNED_DATA = ' + JSON.stringify({ model: rd('model.json'), demo: rd('demo.json'), validation: rd('validation.json') }) + ';\n');
}
