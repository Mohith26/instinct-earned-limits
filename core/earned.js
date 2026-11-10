// Earned Limits core: simulator, risk model, Learn-then-Test certification,
// anytime-valid trust monitor and replay. Pure ES module, no dependencies.
// Runs in the browser (index.html) and in Node (`node --test`).

// ---------- seeded randomness ----------
export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function normal(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
export function poisson(rng, lambda) {
  if (lambda <= 0) return 0;
  if (lambda > 30) return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * normal(rng)));
  const L = Math.exp(-lambda);
  let k = 0, p = 1;
  do { k++; p *= rng(); } while (p > L);
  return k - 1;
}
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
export const logit = (p) => Math.log(p / (1 - p));

// ---------- world spec ----------
// rate: proposals per day for an average-activity user; median: typical $ amount;
// base: logit of "would not approve" for a neutral, user-requested proposal.
export const CATEGORIES = [
  { id: 'groceries', name: 'Groceries', rate: 0.32, median: 112, spread: 0.30, base: -2.0, routine: 0.45, suggested: 0.06, merchants: ['Whole Foods', "Trader Joe's", 'Bi-Rite Market', 'Instacart (Safeway)', 'Rainbow Grocery'] },
  { id: 'pantry', name: 'Pantry & reorders', rate: 0.26, median: 31, spread: 0.35, base: -2.3, routine: 0.65, suggested: 0.10, merchants: ['Amazon', 'Sightglass Coffee', 'Target', 'Grove Co.', 'Thrive Market'] },
  { id: 'food', name: 'Food delivery', rate: 0.38, median: 39, spread: 0.40, base: -1.9, routine: 0.15, suggested: 0.12, merchants: ['DoorDash', 'Uber Eats', 'Caviar', 'Tartine (pickup)', 'Souvla'] },
  { id: 'rides', name: 'Rides', rate: 0.40, median: 24, spread: 0.45, base: -2.6, routine: 0.20, suggested: 0.05, merchants: ['Uber', 'Lyft', 'Waymo'] },
  { id: 'tickets', name: 'Tickets & events', rate: 0.06, median: 128, spread: 0.55, base: -1.2, routine: 0.0, suggested: 0.30, merchants: ['Ticketmaster', 'AXS', 'Eventbrite', 'Resy (deposit)', 'Dice'] },
  { id: 'travel', name: 'Travel', rate: 0.05, median: 410, spread: 0.60, base: -0.9, routine: 0.0, suggested: 0.20, merchants: ['United', 'Alaska Airlines', 'Hotels.com', 'Amtrak', 'Airbnb'] },
  { id: 'gifts', name: 'Gifts & flowers', rate: 0.08, median: 68, spread: 0.45, base: -1.5, routine: 0.0, suggested: 0.35, merchants: ['Farmgirl Flowers', 'Bloomscape', "Uncommon Goods", 'Goldbelly', 'Etsy'] },
  { id: 'clothing', name: 'Clothing & gear', rate: 0.10, median: 92, spread: 0.45, base: -0.9, routine: 0.05, suggested: 0.55, merchants: ['Nike', 'REI', 'Uniqlo', 'Allbirds', 'On Running', 'Everlane'] },
];
export const CAT_INDEX = Object.fromEntries(CATEGORIES.map((c, i) => [c.id, i]));

// Draw a user's latent preferences.
export function drawUser(rng, overrides = {}) {
  const u = {
    activity: Math.exp(0.45 * normal(rng)) * 1.0,
    priceSens: 1.1 + 0.35 * normal(rng),
    suggestAversion: 1.1 + 0.6 * normal(rng),
    newMerchantAversion: 0.8 + 0.3 * normal(rng),
    jumpSens: 4 + 1.0 * normal(rng),
    catOffset: CATEGORIES.map(() => 0.7 * normal(rng)),
    catMedian: CATEGORIES.map((c) => c.median * Math.exp(0.3 * normal(rng))),
    drift: null,
  };
  return Object.assign(u, overrides);
}

// True probability the user would NOT approve (deny or ignore) a proposal.
export function trueRisk(u, p) {
  const c = CATEGORIES[CAT_INDEX[p.cat]];
  let z = c.base + u.catOffset[CAT_INDEX[p.cat]];
  z += u.priceSens * Math.log(p.amount / u.catMedian[CAT_INDEX[p.cat]]);
  if (p.origin === 'request') z -= 1.5;
  if (p.origin === 'routine') z -= 0.6;
  if (p.origin === 'suggested') z += u.suggestAversion;
  if (p.reorder) z -= 0.9;
  if (p.newMerchant) z += u.newMerchantAversion;
  z += u.jumpSens * Math.max(0, p.priceChange - 0.05);
  if (p.timeSensitive) z -= 0.3;
  if (p.group) z += 0.3;
  if (u.drift && p.day >= u.drift.day) {
    z += u.drift.shift;
    if (u.drift.cats && u.drift.cats.includes(p.cat)) z += u.drift.catShift;
  }
  return sigmoid(z);
}

// Generate proposals for days [d0, d1). Each proposal gets a truth label.
export function simulateProposals(rng, u, d0, d1, idPrefix = 'p') {
  const out = [];
  const last = {}; // last purchased amount per (cat, merchant) for reorders
  let n = 0;
  for (let day = d0; day < d1; day++) {
    CATEGORIES.forEach((c, ci) => {
      const k = poisson(rng, c.rate * u.activity);
      for (let j = 0; j < k; j++) {
        const r = rng();
        const origin = r < c.suggested ? 'suggested' : r < c.suggested + c.routine ? 'routine' : 'request';
        const reorder = origin === 'routine' ? rng() < 0.8 : rng() < 0.15;
        const merchantIdx = Math.min(c.merchants.length - 1, Math.floor(Math.pow(rng(), 1.8) * c.merchants.length));
        const merchant = c.merchants[merchantIdx];
        const newMerchant = reorder ? false : rng() < (origin === 'suggested' ? 0.45 : 0.12);
        const key = c.id + '|' + merchant;
        let amount, priceChange = 0;
        if (reorder && last[key]) {
          priceChange = rng() < 0.09 ? 0.15 + 0.45 * rng() : 0.04 * normal(rng);
          amount = last[key] * (1 + priceChange);
        } else {
          amount = u.catMedian[ci] * Math.exp(c.spread * normal(rng));
        }
        amount = Math.max(3, Math.round(amount * 100) / 100);
        const p = {
          id: `${idPrefix}${n++}`, day, cat: c.id, origin, reorder, merchant, newMerchant,
          amount, priceChange: Math.round(priceChange * 1000) / 1000,
          timeSensitive: c.id === 'tickets' ? rng() < 0.5 : c.id === 'clothing' ? rng() < 0.2 : false,
          group: (c.id === 'tickets' || c.id === 'food') ? rng() < 0.25 : false,
        };
        p.pTrue = trueRisk(u, p);
        p.unwanted = rng() < p.pTrue;
        p.response = p.unwanted ? (rng() < 0.3 ? 'dismiss' : 'deny') : 'approve';
        last[key] = amount;
        out.push(p);
      }
    });
  }
  return out;
}

// ---------- features (frozen personal context) ----------
// Personal context is computed ONLY from labelled history before the calibration
// window, so the score function is fixed before any calibration label is seen.
export function personalContext(history, prior) {
  const ctx = { median: [], decline: [], suggestDecline: 0, n: history.length };
  CATEGORIES.forEach((c, ci) => {
    const rows = history.filter((p) => p.cat === c.id);
    const appr = rows.filter((p) => p.response === 'approve').map((p) => p.amount).sort((a, b) => a - b);
    ctx.median[ci] = appr.length >= 2 ? appr[Math.floor(appr.length / 2)] : c.median;
    const d = rows.filter((p) => p.response !== 'approve').length;
    const a = 6, mu = prior ? prior.decline[ci] : 0.15;
    ctx.decline[ci] = logit((d + a * mu) / (rows.length + a));
  });
  const sg = history.filter((p) => p.origin === 'suggested');
  const sd = sg.filter((p) => p.response !== 'approve').length;
  const muS = prior ? prior.suggestDecline : 0.4;
  ctx.suggestDecline = logit((sd + 6 * muS) / (sg.length + 6));
  return ctx;
}

export const FEATURE_NAMES = [
  'intercept', ...CATEGORIES.map((c) => 'cat:' + c.id), 'log amount / your usual', 'you asked for it', 'routine', 'I suggested it',
  'same item as before', 'new store', 'price up vs last time', 'time-sensitive', 'group chat', 'your decline rate here', 'suggested x your suggest-decline rate',
];

export function featurize(p, ctx) {
  const ci = CAT_INDEX[p.cat];
  const x = new Array(FEATURE_NAMES.length).fill(0);
  let k = 0;
  x[k++] = 1;
  for (let i = 0; i < CATEGORIES.length; i++) x[k++] = i === ci ? 1 : 0;
  x[k++] = Math.log(p.amount / ctx.median[ci]);
  x[k++] = p.origin === 'request' ? 1 : 0;
  x[k++] = p.origin === 'routine' ? 1 : 0;
  x[k++] = p.origin === 'suggested' ? 1 : 0;
  x[k++] = p.reorder ? 1 : 0;
  x[k++] = p.newMerchant ? 1 : 0;
  x[k++] = Math.max(0, p.priceChange);
  x[k++] = p.timeSensitive ? 1 : 0;
  x[k++] = p.group ? 1 : 0;
  x[k++] = ctx.decline[ci];
  x[k++] = (p.origin === 'suggested' ? 1 : 0) * ctx.suggestDecline;
  return x;
}

// ---------- L2 logistic regression (Newton / IRLS) ----------
function cholSolve(A, b) {
  const n = b.length, L = A.map((r) => r.slice());
  for (let j = 0; j < n; j++) {
    let s = L[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    L[j][j] = Math.sqrt(Math.max(s, 1e-12));
    for (let i = j + 1; i < n; i++) {
      let t = L[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / L[j][j];
    }
  }
  const y = new Array(n);
  for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  const x = new Array(n);
  for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}
export function fitLogistic(X, y, l2 = 1.0, iters = 30) {
  const d = X[0].length;
  let w = new Array(d).fill(0);
  for (let it = 0; it < iters; it++) {
    const g = new Array(d).fill(0);
    const H = Array.from({ length: d }, () => new Array(d).fill(0));
    for (let i = 0; i < X.length; i++) {
      const xi = X[i];
      let z = 0; for (let j = 0; j < d; j++) z += w[j] * xi[j];
      const p = sigmoid(z), r = p * (1 - p), e = p - y[i];
      for (let j = 0; j < d; j++) {
        g[j] += e * xi[j];
        if (xi[j] === 0) continue;
        const rx = r * xi[j];
        for (let k = j; k < d; k++) H[j][k] += rx * xi[k];
      }
    }
    for (let j = 0; j < d; j++) {
      for (let k = 0; k < j; k++) H[j][k] = H[k][j];
      if (j > 0) { g[j] += l2 * w[j]; H[j][j] += l2; } else H[j][j] += 1e-6;
    }
    const step = cholSolve(H, g);
    let maxStep = 0;
    w = w.map((wj, j) => { maxStep = Math.max(maxStep, Math.abs(step[j])); return wj - step[j]; });
    if (maxStep < 1e-8) break;
  }
  return w;
}
export function score(w, x) { let z = 0; for (let j = 0; j < w.length; j++) z += w[j] * x[j]; return sigmoid(z); }

// ---------- exact binomial tools ----------
function logChoose(n, k) { return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1); }
export function lgamma(z) {
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1; let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
// P(Binom(n, p) <= k)
export function binomCdf(k, n, p) {
  if (k < 0) return 0;
  if (k >= n) return 1;
  if (p <= 0) return 1;
  if (p >= 1) return 0;
  let s = 0;
  for (let i = 0; i <= k; i++) s += Math.exp(logChoose(n, i) + i * Math.log(p) + (n - i) * Math.log(1 - p));
  return Math.min(1, s);
}
// One-sided Clopper-Pearson upper bound for a rate given k events in n trials.
export function cpUpper(k, n, conf = 0.9) {
  if (n === 0) return 1;
  if (k >= n) return 1;
  let lo = k / n, hi = 1;
  for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (binomCdf(k, n, mid) > 1 - conf) lo = mid; else hi = mid; }
  return hi;
}

// ---------- Learn-then-Test certification ----------
// Rule family: auto-buy iff amount <= hardCap AND score <= tau.
// Risk R(tau) = P(auto-bought AND you would not have approved), a marginal rate
// per purchase request, non-decreasing in tau. Fixed-sequence testing over an
// increasing tau grid with exact binomial p-values controls the family-wise
// error at delta, so P(R(tau_hat) > alpha) <= delta (Angelopoulos et al., LTT).
export const TAU_GRID = Array.from({ length: 100 }, (_, i) => Math.round((0.005 + i * 0.005) * 1000) / 1000);
export const ALWAYS_ASK = ['travel'];
// Support cap: never auto-buy more than SUPPORT_MULT x your usual amount in a
// category (r.usual, frozen before calibration), so limits don't extrapolate
// past what your history covers.
export const SUPPORT_MULT = 2.5;
export function eligible(r, hardCap, alwaysAsk = ALWAYS_ASK) {
  return r.amount <= hardCap && !alwaysAsk.includes(r.cat) && (r.usual == null || r.amount <= SUPPORT_MULT * r.usual);
}
// Categories with fewer than minPerCat requests in the window stay on "always ask".
// This uses request counts only (no labels), so it does not break validity.
export function thinCategories(rows, minPerCat = 8) {
  return CATEGORIES.map((c) => c.id).filter((id) => rows.filter((r) => r.cat === id).length < minPerCat);
}
export function certify(rows, { alpha = 0.05, delta = 0.1, hardCap = 300, grid = TAU_GRID, alwaysAsk = ALWAYS_ASK, minPerCat = 8 } = {}) {
  const n = rows.length;
  const thin = minPerCat > 0 ? thinCategories(rows, minPerCat).filter((c) => !alwaysAsk.includes(c)) : [];
  alwaysAsk = alwaysAsk.concat(thin);
  const path = [];
  let tauHat = null;
  for (const tau of grid) {
    let auto = 0, bad = 0;
    for (const r of rows) {
      if (eligible(r, hardCap, alwaysAsk) && r.s <= tau) { auto++; if (r.unwanted) bad++; }
    }
    const pval = binomCdf(bad, n, alpha);
    const ucb = cpUpper(bad, n, 1 - delta);
    path.push({ tau, auto, bad, pval, ucb, rejected: pval <= delta });
    if (pval <= delta) tauHat = tau; else break; // fixed sequence: stop at first failure
  }
  return { tauHat, n, path, alpha, delta, hardCap, alwaysAsk, thin };
}

export function decideAuto(r, cert) { return cert.tauHat != null && eligible(r, cert.hardCap, cert.alwaysAsk || ALWAYS_ASK) && r.s <= cert.tauHat; }

// Largest amount for which a canonical "usual" purchase in a category would be
// auto-bought under tau. Canonical = your usual store, no price change, the way
// you normally buy in that category (routine reorder or your own request).
export function canonical(cat, amount) {
  const c = CATEGORIES[CAT_INDEX[cat]];
  return { cat, amount, origin: c.routine >= 0.4 ? 'routine' : 'request', reorder: c.routine >= 0.4, newMerchant: false, priceChange: 0, timeSensitive: false, group: false };
}
// Which unusual signals would make a usual purchase in this category ask anyway.
export function asksWhen(w, ctx, cat, tau, amount) {
  if (tau == null) return [];
  const base = canonical(cat, amount);
  const tests = [
    ['I suggested it', { origin: 'suggested', reorder: false }],
    ['new store', { newMerchant: true, reorder: false }],
    ['price up 20%+', { priceChange: 0.2 }],
    ['in a group chat', { group: true }],
  ];
  return tests.filter(([, mod]) => score(w, featurize({ ...base, ...mod }, ctx)) > tau).map(([label]) => label);
}
export function effectiveLimit(w, ctx, cat, tau, hardCap, alwaysAsk = ALWAYS_ASK) {
  if (tau == null || alwaysAsk.includes(cat)) return 0;
  hardCap = Math.min(hardCap, Math.floor(SUPPORT_MULT * ctx.median[CAT_INDEX[cat]]));
  const c = CATEGORIES[CAT_INDEX[cat]];
  const base = { cat, origin: c.routine >= 0.4 ? 'routine' : 'request', reorder: c.routine >= 0.4, newMerchant: false, priceChange: 0, timeSensitive: false, group: false };
  const ok = (a) => score(w, featurize({ ...base, amount: a }, ctx)) <= tau;
  if (!ok(1)) return 0;
  let lo = 1, hi = hardCap;
  if (ok(hi)) return hardCap;
  for (let it = 0; it < 40; it++) { const mid = (lo + hi) / 2; if (ok(mid)) lo = mid; else hi = mid; }
  return Math.floor(lo);
}

// Per-feature contributions for "why I'm asking" (logit scale, vs. population mean).
export function reasons(w, x, means) {
  const out = [];
  for (let j = 1; j < w.length; j++) {
    const c = w[j] * (x[j] - (means ? means[j] : 0));
    out.push({ feature: FEATURE_NAMES[j], contribution: c });
  }
  return out.sort((a, b) => b.contribution - a.contribution);
}

// ---------- anytime-valid trust monitor ----------
// Betting e-process for H0: E[Z_t] <= m0 with Z_t >= lo (bounded below).
// K_t = prod (1 + lambda_t (Z_t - m0)), lambda_t predictable in [0, 0.5/(m0 - lo)]
// from a running GRAPA-style estimate. Under H0 K is a nonnegative
// supermartingale, so P(sup_t K_t >= 1/eps) <= eps (Ville's inequality).
// Used on residuals Z = (did not approve) - score: if your answers drift away
// from what the certified model expects, evidence piles up and limits pause.
export function eProcess(values, m0, eps = 0.05, lo = 0, decay = 0.97) {
  // Bet size uses exponentially weighted moments (predictable: only past data),
  // so the bet adapts quickly after a change without breaking validity.
  let K = 1, sum = 0, sumsq = 0, n = 0;
  const path = [];
  let tripped = null;
  const lamMax = 0.5 / Math.max(m0 - lo, 1e-6);
  for (let t = 0; t < values.length; t++) {
    const mhat = (sum + m0) / (n + 1);
    const vhat = Math.max(1e-4, (sumsq + m0 * m0) / (n + 1) - mhat * mhat);
    const lam = Math.max(0, Math.min(lamMax, (mhat - m0) / (vhat + (mhat - m0) ** 2)));
    const z = typeof values[t] === 'boolean' ? (values[t] ? 1 : 0) : values[t];
    K *= 1 + lam * (z - m0);
    sum = decay * sum + z; sumsq = decay * sumsq + z * z; n = decay * n + 1;
    path.push(K);
    if (tripped == null && K >= 1 / eps) tripped = t;
  }
  return { K, path, tripped, threshold: 1 / eps };
}

// Null level for the residual monitor: calibration-window mean residual plus a
// one-sided 95% normal margin (so an honest user rarely starts above it).
export function residualNull(rows) {
  const z = rows.map((r) => (r.unwanted ? 1 : 0) - r.s);
  const n = z.length, m = z.reduce((a, b) => a + b, 0) / n;
  const sd = Math.sqrt(z.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, n - 1));
  return { mean: m, sd, n, m0: m + 1.645 * sd / Math.sqrt(n) };
}

// Conditional unwanted-rate bound among auto-bought calibration rows (used as q0).
export function conditionalBound(rows, tau, hardCap, conf = 0.9) {
  const a = rows.filter((r) => r.amount <= hardCap && r.s <= tau);
  const b = a.filter((r) => r.unwanted).length;
  return { n: a.length, bad: b, upper: cpUpper(b, a.length, conf) };
}

// ---------- replay ----------
export function replay(rows, decide) {
  let auto = 0, asked = 0, autoBad = 0, autoBadDollars = 0, autoDollars = 0;
  for (const r of rows) {
    if (decide(r)) { auto++; autoDollars += r.amount; if (r.unwanted) { autoBad++; autoBadDollars += r.amount; } }
    else asked++;
  }
  return { total: rows.length, auto, asked, autoBad, autoBadDollars: Math.round(autoBadDollars * 100) / 100, autoDollars: Math.round(autoDollars * 100) / 100 };
}

// ---------- population prior + training ----------
export function populationPrior(rows) {
  const decline = CATEGORIES.map((c) => {
    const r = rows.filter((p) => p.cat === c.id);
    return r.length ? r.filter((p) => p.response !== 'approve').length / r.length : 0.15;
  });
  const sg = rows.filter((p) => p.origin === 'suggested');
  return { decline, suggestDecline: sg.length ? sg.filter((p) => p.response !== 'approve').length / sg.length : 0.4 };
}
