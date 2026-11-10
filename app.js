import {
  CATEGORIES, CAT_INDEX, ALWAYS_ASK, personalContext, featurize, score, certify, decideAuto,
  effectiveLimit, asksWhen, replay, eProcess, residualNull, simulateProposals, mulberry32,
} from './core/earned.js';

const { model, demo, validation } = window.EARNED_DATA;
const W = model.weights;
const T = demo.timeline;
const S = demo.settings;
const DELTA = S.delta;
const START = new Date(T.start + 'T12:00:00');

const $ = (id) => document.getElementById(id);
const dateOf = (day) => new Date(START.getTime() + day * 86400000);
const fmtDate = (day) => dateOf(day).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
const usd = (v) => '$' + v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const usd0 = (v) => '$' + Math.round(v).toLocaleString('en-US');
const pct = (v, d = 0) => (v * 100).toFixed(d) + '%';
const oneIn = (a) => '1 in ' + Math.round(1 / a);

// ----- frozen personal context and scored rows -----
const P = demo.proposals;
const ctx = personalContext(P.filter((p) => p.day < T.warmupEnd), model.prior);
const scoreRow = (p) => ({ ...p, s: score(W, featurize(p, ctx)), usual: ctx.median[CAT_INDEX[p.cat]] });
const calib = P.filter((p) => p.day >= T.warmupEnd && p.day < T.calibEnd).map(scoreRow);
const recent = P.filter((p) => p.day >= T.calibEnd).map(scoreRow);
const nul = residualNull(calib);

const ICONS = { groceries: 'shopping-basket', pantry: 'coffee', food: 'utensils', rides: 'car', tickets: 'ticket', travel: 'plane', gifts: 'gift', clothing: 'shirt' };

let alpha = S.alpha;
let cert = null;
let simState = null;

function whyAsks(r, lim) {
  const out = [];
  if (ALWAYS_ASK.includes(r.cat)) return ['travel always asks'];
  if (cert && cert.thin.includes(r.cat)) return ['still learning this category'];
  if (r.amount > S.hardCap) return ['over your $' + S.hardCap + ' hard limit'];
  if (r.origin === 'suggested') out.push('I suggested it');
  if (r.newMerchant) out.push('new store for you');
  if (r.priceChange > 0.12) out.push('price up ' + Math.round(r.priceChange * 100) + '% since last time');
  if (lim != null && r.amount > lim) out.push('above your earned ' + usd0(lim));
  if (r.group) out.push('from a group chat');
  if (!out.length) out.push('looks unusual for you (score ' + pct(r.s, 1) + ')');
  return out;
}
function outcomeOf(r, wasAuto) {
  if (wasAuto) return r.unwanted ? { text: 'You returned it', cls: 'bad' } : { text: 'You kept it', cls: '' };
  if (r.response === 'approve') return { text: 'You approved', cls: '' };
  return r.response === 'deny' ? { text: 'You declined', cls: 'bad' } : { text: 'You ignored it', cls: 'bad' };
}

function render() {
  cert = certify(calib, { alpha, delta: DELTA, hardCap: S.hardCap });
  const last = cert.path.filter((p) => p.rejected).at(-1);
  const limits = {};
  CATEGORIES.forEach((c) => { limits[c.id] = effectiveLimit(W, ctx, c.id, cert.tauHat, S.hardCap, cert.alwaysAsk); });

  // Link balance
  const flatDecide = (r) => r.amount <= S.staticLimit;
  const flat = replay(recent, flatDecide);
  $('balanceValue').textContent = usd(flat.autoDollars);
  $('balanceMeta').textContent = `Spent since ${fmtDate(T.calibEnd)} on ${flat.auto} purchases made without asking. Your per-purchase limit is ${usd0(S.staticLimit)}.`;
  $('hardCapValue').textContent = usd0(S.hardCap);

  // summary
  const earnedDecide = (r) => decideAuto(r, cert);
  const earned = replay(recent, earnedDecide);
  $('statAuto').innerHTML = `${earned.auto}<span>/${earned.total}</span>`;
  $('statAutoLabel').textContent = `requests since ${fmtDate(T.calibEnd)} would skip the approval text · ${earned.autoBad} unwanted (flat ${usd0(S.staticLimit)} limit: ${flat.autoBad})`;
  $('meter').innerHTML = recent.map((r) => `<span class="${earnedDecide(r) ? (r.unwanted ? 'b' : 'a') : ''}" title="${fmtDate(r.day)} · ${r.merchant} · ${usd(r.amount)}"></span>`).join('');
  $('statRisk').textContent = cert.tauHat == null ? '–' : `${pct(last.ucb, 1)}`;
  $('statRiskLabel').textContent = cert.tauHat == null
    ? `Not enough answers yet to certify ${oneIn(alpha)}.`
    : `Upper bound on purchase requests that end in a no-ask buy you wouldn't want: under your ${oneIn(alpha)} target with ${pct(1 - DELTA)} confidence.`;
  $('statEvidence').textContent = `${cert.n}`;
  $('statEvidenceLabel').textContent = `Your yes and no replies from ${fmtDate(T.warmupEnd)} to ${fmtDate(T.calibEnd - 1)}, before Link balance, when Instinct asked about every purchase.`;

  document.querySelectorAll('#alphaSeg button').forEach((b) => b.setAttribute('aria-checked', String(Number(b.dataset.alpha) === alpha)));

  // category rows
  const rows = CATEGORIES.map((c) => {
    const ev = calib.filter((r) => r.cat === c.id);
    const no = ev.filter((r) => r.unwanted).length;
    const noSug = ev.filter((r) => r.unwanted && r.origin === 'suggested').length;
    const evidence = `${ev.length} answer${ev.length === 1 ? '' : 's'}, ${no} no` + (no > 0 && noSug === no ? (no === 1 ? ' (on something I suggested)' : ' (all on things I suggested)') : noSug > 0 ? ` (${noSug} on things I suggested)` : '');
    const lim = limits[c.id];
    let chip, meta, val;
    if (ALWAYS_ASK.includes(c.id)) {
      chip = '<span class="chip always">Always asks</span>';
      meta = `Travel is hard to undo, so it always waits for you. ${evidence}.`;
      val = '<span class="value muted">Asks</span>';
    } else if (cert.thin.includes(c.id)) {
      chip = '<span class="chip asking">Learning</span>';
      meta = `Needs at least 8 requests here before a limit can be earned. ${evidence}.`;
      val = '<span class="value muted">Asks</span>';
    } else if (lim >= 5) {
      const usual = ctx.median[CAT_INDEX[c.id]];
      const aw = asksWhen(W, ctx, c.id, cert.tauHat, Math.min(usual, lim));
      const capNote = lim >= S.hardCap ? ` up to your ${usd0(S.hardCap)} hard limit` : ` up to ${usd0(lim)}`;
      const usualNote = ` (you usually spend about ${usd0(usual)})`;
      meta = `Usual orders${capNote} without a text${usualNote}.` + (aw.length ? ` Still asks if ${aw.join(', ')}.` : '') + ` ${evidence}.`;
      chip = '<span class="chip earned">Earned</span>';
      val = `<span class="value">${usd0(lim)}</span>`;
    } else {
      chip = '<span class="chip asking">Asking</span>';
      meta = `Your answers here don't support skipping the text yet. ${evidence}.`;
      val = '<span class="value muted">Asks</span>';
    }
    return `<div class="row"><span class="row-icon"><i data-lucide="${ICONS[c.id]}"></i></span>
      <div class="row-body"><span class="row-title">${c.name} ${chip}</span><span class="row-meta">${meta}</span></div>
      <div class="row-end">${val}</div></div>`;
  });
  $('catRows').innerHTML = rows.join('');

  // replay
  $('replayDesc').textContent = `Your last ${T.replayEnd - T.calibEnd} days (${fmtDate(T.calibEnd)} to ${fmtDate(T.replayEnd - 1)}) played through both policies. None of these requests were used to earn the limits.`;
  const bar = (r) => `<div class="bar"><i style="width:${(100 * r.auto / r.total).toFixed(1)}%"></i><b style="left:${(100 * (r.auto - r.autoBad) / r.total).toFixed(1)}%;width:${(100 * r.autoBad / r.total).toFixed(1)}%"></b></div>`;
  $('replayRows').innerHTML = `
    <div class="row align-start"><span class="row-icon"><i data-lucide="equal"></i></span>
      <div class="row-body"><span class="row-title">Flat ${usd0(S.staticLimit)} limit <span class="chip asking">Today</span></span>
      <span class="row-meta">Bought ${flat.auto} of ${flat.total} without asking. ${flat.autoBad} of those you returned or said you didn't want (${usd(flat.autoBadDollars)}).</span>${bar(flat)}</div>
      <div class="row-end"><span class="value">${flat.autoBad} unwanted</span></div></div>
    <div class="row align-start"><span class="row-icon"><i data-lucide="sparkles"></i></span>
      <div class="row-body"><span class="row-title">Earned limits <span class="chip earned">${oneIn(alpha)}</span></span>
      <span class="row-meta">Would have bought ${earned.auto} of ${earned.total} without asking. ${earned.autoBad} unwanted (${usd(earned.autoBadDollars)}).</span>${bar(earned)}</div>
      <div class="row-end"><span class="value">${earned.autoBad} unwanted</span></div></div>`;

  const diffs = recent.filter((r) => flatDecide(r) !== earnedDecide(r));
  $('diffSummary').textContent = `${diffs.length} requests decided differently`;
  $('diffList').innerHTML = diffs.map((r) => {
    const cat = CATEGORIES[CAT_INDEX[r.cat]].name;
    const earnedAsks = flatDecide(r) && !earnedDecide(r);
    const lim = limits[r.cat];
    const why = earnedAsks ? 'Earned limits would ask: ' + whyAsks(r, lim).join(', ') : 'Earned limits would buy it without a text';
    const o = outcomeOf(r, flatDecide(r));
    let cls = o.cls;
    if (earnedAsks) cls = r.unwanted ? 'good' : '';
    else cls = r.unwanted ? 'bad' : 'good';
    const tag = earnedAsks ? (r.unwanted ? ' · caught' : ' · extra text') : (r.unwanted ? ' · would be a mistake' : ' · one less text');
    return `<div class="diff"><span class="d-date">${fmtDate(r.day)}</span>
      <span class="d-what">${r.merchant} · ${cat} · ${usd(r.amount)}<span class="d-why">${why}</span></span>
      <span class="d-out ${cls}">${o.text}${tag}</span></div>`;
  }).join('');

  renderTrust();
  renderHow(last);
  if (window.lucide) window.lucide.createIcons({ attrs: { 'stroke-width': 1.5 } });
}

function sparkPath(path, startDay, endDay, days, threshold, markDay) {
  const svg = $('spark');
  const Wd = 600, H = 120, lo = Math.log10(0.25), hi = Math.log10(60);
  const x = (d) => ((d - startDay) / Math.max(1, endDay - startDay)) * Wd;
  const y = (k) => H - ((Math.min(hi, Math.max(lo, Math.log10(k))) - lo) / (hi - lo)) * H;
  let d = `M0 ${y(1).toFixed(1)}`;
  path.forEach((k, i) => { d += ` L${x(days[i]).toFixed(1)} ${y(k).toFixed(1)}`; });
  const ty = y(threshold).toFixed(1), oy = y(1).toFixed(1);
  svg.innerHTML = `
    <line x1="0" x2="${Wd}" y1="${oy}" y2="${oy}" stroke="rgb(77 53 51 / .1)" stroke-dasharray="2 3"/>
    <line x1="0" x2="${Wd}" y1="${ty}" y2="${ty}" stroke="#e36d42" stroke-opacity=".55" stroke-dasharray="4 3"/>
    <text x="${Wd - 4}" y="${Number(ty) - 5}" text-anchor="end" font-size="11" fill="#c95f5b">pause at ${threshold}×</text>
    <text x="4" y="${Number(oy) - 5}" font-size="11" fill="rgb(77 53 51 / .5)">1×</text>
    ${markDay != null ? `<line x1="${x(markDay).toFixed(1)}" x2="${x(markDay).toFixed(1)}" y1="14" y2="${H}" stroke="rgb(77 53 51 / .25)" stroke-dasharray="2 3"/><text x="${(x(markDay) + 4).toFixed(1)}" y="24" font-size="11" fill="rgb(77 53 51 / .5)">habits change</text>` : ''}
    <path d="${d}" fill="none" stroke="#4d3533" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/>`;
}

function renderTrust() {
  const base = recent.map((r) => (r.unwanted ? 1 : 0) - r.s);
  const baseDays = recent.map((r) => r.day);
  let vals = base, days = baseDays, endDay = T.replayEnd - 1, extra = null;
  if (simState) {
    vals = base.concat(simState.vals); days = baseDays.concat(simState.days); endDay = simState.endDay; extra = simState;
  }
  const ep = eProcess(vals, nul.m0, 0.05, -1);
  const unwanted = recent.filter((r) => r.unwanted).length, expected = recent.reduce((a, r) => a + r.s, 0);
  let shown = ep.path, shownDays = days;
  if (ep.tripped != null) { shown = ep.path.slice(0, ep.tripped + 1); shownDays = days.slice(0, ep.tripped + 1); }
  if (ep.tripped == null) {
    $('trustTitle').innerHTML = `Healthy <span class="chip earned">Earned limits on</span>`;
    $('trustMeta').textContent = extra
      ? `Simulated ${extra.n} more requests with no change in habits. Evidence of a change: ${ep.K.toFixed(2)}× (pauses at ${ep.threshold}×).`
      : `Since ${fmtDate(T.calibEnd)}: ${unwanted} unwanted out of ${recent.length} requests, where the model expected ${expected.toFixed(1)}. Evidence of a change: ${ep.K.toFixed(2)}× (pauses at ${ep.threshold}×).`;
  } else {
    const d = days[ep.tripped];
    $('trustTitle').innerHTML = `Paused on ${fmtDate(d)} <span class="chip paused">Asking for everything</span>`;
    $('trustMeta').textContent = `Simulated change on ${fmtDate(extra.driftDay)}: you start declining more (a new budget), mostly food, gifts and clothing. Evidence passed ${ep.threshold}× ${d - extra.driftDay} days into the new habit (${ep.tripped + 1 - base.length} requests after ${fmtDate(T.replayEnd - 1)}). Earned limits stay paused until new answers re-earn them.`;
  }
  sparkPath(shown, T.calibEnd, endDay, shownDays, ep.threshold, extra ? extra.driftDay : null);
  $('sparkStart').textContent = fmtDate(T.calibEnd);
  $('sparkEnd').textContent = fmtDate(endDay);
  $('simBtn').textContent = simState ? 'Reset' : 'Simulate a change in habits';
}

function runSim() {
  const rng = mulberry32(424242);
  const user = { ...demo.user, drift: { day: T.replayEnd + 7, shift: 1.2, cats: ['food', 'clothing', 'gifts'], catShift: 1.0 } };
  const props = simulateProposals(rng, user, T.replayEnd, T.replayEnd + 60, 's').map(scoreRow);
  simState = { vals: props.map((r) => (r.unwanted ? 1 : 0) - r.s), days: props.map((r) => r.day), endDay: T.replayEnd + 59, n: props.length, driftDay: user.drift.day };
  renderTrust();
}

function renderHow(last) {
  const v = validation;
  const rows = [
    ['gauge', 'A risk score for every purchase request',
      `A logistic model trained on ${model.trainUsers} simulated users scores how likely you are to say no, from the amount against your usual, who asked for it, the store, price changes and your own past answers. On a fresh cohort it ranks declines with AUC ${v.model.auc.toFixed(2)}.`],
    ['list-checks', 'Certified threshold (Learn then Test)',
      cert.tauHat == null ? 'No threshold passed yet, so Instinct keeps asking for everything.' :
      `Thresholds are tested from strictest to loosest with an exact binomial test on your ${cert.n} answers, stopping at the first one that fails. The loosest that passes is ${pct(cert.tauHat, 1)}: ${last.auto} of your ${cert.n} past requests fall under it, ${last.bad} of them ones you turned down.`],
    ['flask-conical', `Checked on ${v.users} more simulated users`,
      `Kept the true rate of unwanted no-ask buys at or under 1 in 20 for ${pct(v.coverage, 1)} of the ${Math.round(v.earnedShare * v.users)} users who earned a limit (the math promises at least 90%). A flat $${v.staticLimit} limit broke that target for ${pct(v.staticOverAlpha)} of users. Certified the same way, a flat dollar limit skipped ${pct(v.medianAutoFlatCertified)} of approval texts for the median user; earned limits skipped ${pct(v.medianAutoEarnedAll)}.`],
    ['activity', 'Pausing when habits change',
      `The trust check is a betting test that stays valid however often it is checked. With no change it paused ${pct(v.monitor.falseAlarmRate)} of users over 120 days; after a simulated change in habits it paused ${pct(v.monitor.detectRate, 1)}, a median of ${v.monitor.medianDelayDays} days later.`],
    ['info', 'What it can\u2019t promise',
      `The guarantee assumes your next requests look like your past ones, which is why the trust check exists. Accounts need about ${v.minCalibRowsEarned} answers before any limit is earned, and returns are an imperfect signal for no-ask buys you quietly regret.`],
  ];
  $('howRows').innerHTML = rows.map(([icon, t, m]) => `<div class="row align-start"><span class="row-icon"><i data-lucide="${icon}"></i></span>
    <div class="row-body"><span class="row-title">${t}</span><span class="row-meta">${m}</span></div></div>`).join('');
}

function openPreview() {
  const limits = {};
  CATEGORIES.forEach((c) => { limits[c.id] = effectiveLimit(W, ctx, c.id, cert.tauHat, S.hardCap, cert.alwaysAsk); });
  // the most useful example: a request the flat limit auto-bought, earned limits would ask about, and you returned
  const pick = recent.filter((r) => r.amount <= S.staticLimit && !decideAuto(r, cert) && r.unwanted).sort((a, b) => b.amount - a.amount)[0]
    || recent.filter((r) => !decideAuto(r, cert)).sort((a, b) => b.s - a.s)[0];
  const cat = CATEGORIES[CAT_INDEX[pick.cat]].name.toLowerCase();
  const why = whyAsks(pick, limits[pick.cat]);
  const lead = pick.origin === 'suggested' ? `I put together a ${cat} order from ${pick.merchant}` : `Your ${cat} order from ${pick.merchant} is ready`;
  $('thread').innerHTML = `
    <div class="bubble in">${lead}, ${usd(pick.amount)}.</div>
    <div class="bubble in">Checking with you first since ${why.join(' and ')}. Tap to approve: <a href="#">app.instinct.com/approve</a></div>`;
  $('approveCard').innerHTML = `
    <div class="approve-q">Buy from <b>${pick.merchant}</b> for <b>${usd(pick.amount)}</b>?</div>
    <div class="approve-why">Earned limits cover your usual ${cat} up to ${usd0(limits[pick.cat])}. This one is different: ${why.join(', ')}.</div>
    <button class="btn primary" type="button">Allow once</button>
    <button class="btn secondary" type="button">Always allow at ${pick.merchant}</button>
    <button class="btn secondary" type="button">Don\u2019t allow</button>`;
  $('scrim').hidden = false;
  if (window.lucide) window.lucide.createIcons({ attrs: { 'stroke-width': 1.5 } });
}

document.querySelectorAll('#alphaSeg button').forEach((b) => b.addEventListener('click', () => { alpha = Number(b.dataset.alpha); render(); }));
$('simBtn').addEventListener('click', () => { if (simState) { simState = null; renderTrust(); } else runSim(); });
$('earnedSwitch').addEventListener('click', (e) => {
  const on = !e.currentTarget.classList.contains('on');
  e.currentTarget.classList.toggle('on', on);
  e.currentTarget.setAttribute('aria-checked', String(on));
  document.getElementById('earned').style.opacity = on ? '' : '0.55';
});
$('previewBtn').addEventListener('click', openPreview);
$('dlgClose').addEventListener('click', () => { $('scrim').hidden = true; });
$('scrim').addEventListener('click', (e) => { if (e.target.id === 'scrim') $('scrim').hidden = true; });
render();
// Deep links for review: #preview opens the approval text, #sim runs the habit-change simulation, #diff opens the list.
if (location.hash === '#preview') openPreview();
if (location.hash === '#sim') runSim();
if (location.hash === '#diff') $('diffDetails').open = true;
