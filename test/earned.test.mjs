import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mulberry32, binomCdf, cpUpper, fitLogistic, score, sigmoid, certify, decideAuto, eProcess, residualNull,
  drawUser, simulateProposals, personalContext, featurize, populationPrior, effectiveLimit, thinCategories, CATEGORIES, CAT_INDEX,
} from '../core/earned.js';

test('binomial CDF matches closed forms', () => {
  assert.ok(Math.abs(binomCdf(0, 10, 0.1) - 0.9 ** 10) < 1e-12);
  assert.ok(Math.abs(binomCdf(1, 5, 0.5) - 6 / 32) < 1e-12);
  assert.equal(binomCdf(10, 10, 0.3), 1);
  assert.equal(binomCdf(-1, 10, 0.3), 0);
});

test('Clopper-Pearson upper bound: zero events gives 1-(1-conf)^(1/n)', () => {
  const u = cpUpper(0, 50, 0.9);
  assert.ok(Math.abs(u - (1 - 0.1 ** (1 / 50))) < 1e-6);
  assert.ok(cpUpper(5, 50, 0.9) > 0.1);
});

test('logistic regression recovers known weights', () => {
  const rng = mulberry32(3);
  const wTrue = [-1, 2, -1.5];
  const X = [], y = [];
  for (let i = 0; i < 6000; i++) {
    const x = [1, rng() * 2 - 1, rng() * 2 - 1];
    X.push(x); y.push(rng() < sigmoid(wTrue[0] + wTrue[1] * x[1] + wTrue[2] * x[2]) ? 1 : 0);
  }
  const w = fitLogistic(X, y, 0.01);
  w.forEach((wj, j) => assert.ok(Math.abs(wj - wTrue[j]) < 0.2, `w${j}=${wj}`));
});

test('certify stops at the first threshold that fails (fixed sequence)', () => {
  // scores in increasing order; unwanted only at high scores
  const rows = Array.from({ length: 200 }, (_, i) => ({ cat: 'groceries', amount: 50, s: i / 200, unwanted: i >= 60 }));
  const c = certify(rows, { alpha: 0.05, delta: 0.1, hardCap: 300, minPerCat: 0 });
  assert.ok(c.tauHat != null);
  const firstFail = c.path.find((p) => !p.rejected);
  assert.ok(firstFail.tau > c.tauHat);
  assert.equal(c.path.at(-1), firstFail);
  // every accepted step is a valid binomial rejection
  c.path.filter((p) => p.rejected).forEach((p) => assert.ok(binomCdf(p.bad, 200, 0.05) <= 0.1));
});

test('certify refuses with too little evidence', () => {
  const rows = Array.from({ length: 30 }, () => ({ cat: 'groceries', amount: 20, s: 0.01, unwanted: false }));
  const c = certify(rows, { alpha: 0.05, delta: 0.1, minPerCat: 0 });
  assert.equal(c.tauHat, null); // 0.95^30 = 0.21 > 0.1
});

test('hard cap, always-ask and thin categories are never auto-bought', () => {
  const rows = [
    ...Array.from({ length: 100 }, () => ({ cat: 'groceries', amount: 20, s: 0.01, unwanted: false })),
    { cat: 'travel', amount: 50, s: 0.0, unwanted: false },
    { cat: 'tickets', amount: 50, s: 0.0, unwanted: false },
  ];
  const c = certify(rows, { alpha: 0.05, delta: 0.1, hardCap: 300 });
  assert.ok(c.thin.includes('tickets'));
  assert.equal(decideAuto({ cat: 'groceries', amount: 301, s: 0 }, c), false);
  assert.equal(decideAuto({ cat: 'travel', amount: 20, s: 0 }, c), false);
  assert.equal(decideAuto({ cat: 'tickets', amount: 20, s: 0 }, c), false);
  assert.equal(decideAuto({ cat: 'groceries', amount: 20, s: 0, usual: 5 }, c), false); // > 2.5x usual
  assert.equal(decideAuto({ cat: 'groceries', amount: 20, s: 0, usual: 20 }, c), true);
});

test('thinCategories uses counts only', () => {
  const rows = Array.from({ length: 8 }, () => ({ cat: 'rides' }));
  const thin = thinCategories(rows, 8);
  assert.ok(!thin.includes('rides'));
  assert.ok(thin.includes('groceries'));
});

test('LTT guarantee holds across many simulated users (P(risk > alpha) <= delta)', () => {
  const rng = mulberry32(11);
  const train = Array.from({ length: 150 }, () => drawUser(rng));
  const tp = train.map((u, i) => simulateProposals(rng, u, 0, 77, 't' + i));
  const prior = populationPrior(tp.flat().filter((p) => p.day < 28));
  const X = [], y = [];
  tp.forEach((ps) => { const ctx = personalContext(ps.filter((p) => p.day < 28), prior); ps.filter((p) => p.day >= 28).forEach((p) => { X.push(featurize(p, ctx)); y.push(p.unwanted ? 1 : 0); }); });
  const w = fitLogistic(X, y, 1);
  let earned = 0, bad = 0;
  for (let i = 0; i < 80; i++) {
    const u = drawUser(rng, { activity: 2.2 });
    const ps = simulateProposals(rng, u, 0, 63, 'v' + i);
    const ctx = personalContext(ps.filter((p) => p.day < 28), prior);
    const cal = ps.filter((p) => p.day >= 28).map((p) => ({ ...p, s: score(w, featurize(p, ctx)), usual: ctx.median[CAT_INDEX[p.cat]] }));
    const c = certify(cal, { alpha: 0.05, delta: 0.1 });
    if (c.tauHat == null) continue;
    earned++;
    const fresh = simulateProposals(mulberry32(1000 + i), u, 63, 463, 'f');
    let r = 0;
    fresh.forEach((p) => { if (decideAuto({ ...p, s: score(w, featurize(p, ctx)), usual: ctx.median[CAT_INDEX[p.cat]] }, c)) r += p.pTrue; });
    if (r / fresh.length > 0.05) bad++;
  }
  assert.ok(earned >= 30, 'enough users earned limits: ' + earned);
  assert.ok(bad / earned <= 0.1, `violation rate ${bad}/${earned}`);
});

test('e-process: rarely trips under the null, trips under a clear shift', () => {
  let falseTrips = 0;
  for (let s = 0; s < 300; s++) {
    const rng = mulberry32(500 + s);
    const z = Array.from({ length: 300 }, () => (rng() < 0.1 ? 1 : 0) - 0.1); // mean 0
    if (eProcess(z, 0.0, 0.05, -1).tripped != null) falseTrips++;
  }
  assert.ok(falseTrips / 300 <= 0.05, 'false trips ' + falseTrips);
  const rng = mulberry32(9);
  const z = Array.from({ length: 300 }, () => (rng() < 0.35 ? 1 : 0) - 0.1); // mean 0.25
  assert.ok(eProcess(z, 0.0, 0.05, -1).tripped != null);
});

test('residualNull sits above the calibration mean', () => {
  const rows = Array.from({ length: 100 }, (_, i) => ({ s: 0.2, unwanted: i % 5 === 0 }));
  const n = residualNull(rows);
  assert.ok(Math.abs(n.mean) < 1e-9);
  assert.ok(n.m0 > n.mean);
});

test('effective limit grows with tau and respects the support cap', () => {
  const ctx = { median: CATEGORIES.map((c) => c.median), decline: CATEGORIES.map(() => -2), suggestDecline: -0.5 };
  const w = new Array(22).fill(0); w[0] = -3; w[9] = 1.0; // intercept, log amount ratio
  const a = effectiveLimit(w, ctx, 'groceries', 0.05, 300);
  const b = effectiveLimit(w, ctx, 'groceries', 0.2, 300);
  assert.ok(b >= a);
  assert.ok(b <= Math.floor(2.5 * ctx.median[0]));
  assert.equal(effectiveLimit(w, ctx, 'travel', 0.5, 300), 0);
});

test('simulator is deterministic per seed', () => {
  const a = simulateProposals(mulberry32(5), drawUser(mulberry32(1)), 0, 10, 'x');
  const b = simulateProposals(mulberry32(5), drawUser(mulberry32(1)), 0, 10, 'x');
  assert.deepEqual(a, b);
});
