const $ = s => document.querySelector(s);
const $$ = s => [...document.querySelectorAll(s)];
const state = {
  year: null,
  crime: 'Total Crime',
  state: 'All',
  mapMetric: 'cri',
  map: null,
  spatialMap: null,
  geo: null,
  mapLayers: [],
  growthLegends: { map: null, mapSpatial: null },
  spatialLayers: [],
  forecastHorizon: 3,
  mlModel: 'ensemble',
  mlMode: 'future',
  mlHorizon: 3,
  mlToken: 0,
  temporalStart: null,
  temporalEnd: null,
  forecastOrigin: null,
  heatMode: 'normalized',
  heatSort: 'severity',
  renderToken: 0,
  forecastToken: 0,
  refreshToken: 0,
};

// Try several public boundary sources. The first source is preferred; later sources are fallbacks.
const geoUrls = [
  '/api/geojson',
  'https://raw.githubusercontent.com/india-in-data/india-states-2019/master/india_states.geojson',
  'https://cdn.jsdelivr.net/gh/AbhinavSwami28/india-official-geojson@main/india-states-simplified.geojson',
  'https://raw.githubusercontent.com/AbhinavSwami28/india-official-geojson/main/india-states-simplified.geojson',
  'https://webgis1.nic.in/nicstreet/rest/services/terraintile/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson'
];

const plotCfg = { responsive: true, displayModeBar: false };

function fmt(n, digits = 1) {
  if (n == null || Number.isNaN(+n) || !Number.isFinite(+n)) return '—';
  return new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(+n);
}

function toast(msg) {
  const e = $('#toast');
  e.textContent = msg;
  e.classList.add('show');
  setTimeout(() => e.classList.remove('show'), 2800);
}

async function api(path) {
  const r = await fetch(path);
  const text = await r.text();
  let d;
  try { d = JSON.parse(text); } catch { throw new Error('Server returned invalid JSON. Check the backend console.'); }
  if (!r.ok || d.error) throw new Error(d.error || 'Request failed');
  return d;
}

function invalidateVisiblePlots() {
  const ids = ['trendChart', 'categoryChart', 'temporalChart', 'heatmap', 'spatialBar', 'criChart', 'forecastChart', 'mlForecastChart'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.Plotly && el.data) {
      try { Plotly.Plots.resize(el); } catch (_) { }
    }
  });
  if (state.map) { try { state.map.invalidateSize(true); } catch (_) { } }
  if (state.spatialMap) { try { state.spatialMap.invalidateSize(true); } catch (_) { } }
}

async function init() {
  bind();
  try {
    const m = await api('/api/meta');
    populateMeta(m);
    await loadGeo();
    await refreshAll();
  } catch (e) {
    toast(e.message);
  }
}

function bind() {
  $('#uploadBtn').onclick = () => $('#fileInput').click();
  $('#fileInput').onchange = async e => {
    const file = e.target.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append('file', file);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const text = await r.text();
      let d;
      try { d = JSON.parse(text); } catch { throw new Error('Upload returned invalid JSON.'); }
      if (!r.ok || d.error) throw new Error(d.error || 'Upload failed');
      toast(`Loaded ${d.source}: ${d.rows.toLocaleString()} normalized rows`);
      const m = await api('/api/meta');
      populateMeta(m);
      await refreshAll();
      e.target.value = '';
    } catch (err) {
      e.target.value = '';
      toast(err.message);
    }
  };

  $$('#nav .nav').forEach(b => b.onclick = () => switchView(b.dataset.view));
  $('#yearSelect').onchange = async e => {
    state.year = +e.target.value;
    if ($('#view-forecast').classList.contains('active-view')) {
      state.forecastOrigin = state.year;
      $('#forecastOriginSelect').value = state.forecastOrigin;
      await refreshForecast();
    } else {
      await refreshAll();
    }
  };
  $('#forecastOriginSelect').onchange = async e => {
    state.forecastOrigin = +e.target.value;
    await refreshForecast();
  };
  $('#startYearSelect').onchange = async e => {
    state.temporalStart = +e.target.value;
    if (state.temporalStart > state.temporalEnd) {
      state.temporalEnd = state.temporalStart;
      $('#endYearSelect').value = state.temporalEnd;
    }
    if ($('#view-temporal').classList.contains('active-view')) await refreshTemporal();
  };
  $('#endYearSelect').onchange = async e => {
    state.temporalEnd = +e.target.value;
    if (state.temporalEnd < state.temporalStart) {
      state.temporalStart = state.temporalEnd;
      $('#startYearSelect').value = state.temporalStart;
    }
    if ($('#view-temporal').classList.contains('active-view')) await refreshTemporal();
  };
  $('#crimeSelect').onchange = async e => {
    state.crime = e.target.value;
    await refreshCurrent();
  };
  $('#stateSelect').onchange = async e => {
    state.state = e.target.value;
    if ($('#view-forecast').classList.contains('active-view')) {
      await refreshForecastOptions();
      await refreshForecast();
    } else if ($('#view-ml').classList.contains('active-view')) {
      await refreshMl();
    } else {
      await refreshCurrent();
    }
  };
  $$('#mlModeSeg button').forEach(b => b.onclick = async () => {
    $$('#mlModeSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.mlMode = b.dataset.mode;
    const isVal = state.mlMode === 'validation';
    $('#mlHorizonLabel').classList.toggle('hidden', isVal);
    if ($('#view-ml').classList.contains('active-view')) await refreshMl();
  });
  $$('#mlModelSeg button').forEach(b => b.onclick = async () => {
    $$('#mlModelSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.mlModel = b.dataset.model;
    if ($('#view-ml').classList.contains('active-view')) await refreshMl();
  });
  const onMlHorizonChange = async e => {
    state.mlHorizon = clampInt(e.target.value, 1, 20, 3);
    e.target.value = state.mlHorizon;
    if ($('#view-ml').classList.contains('active-view')) await refreshMl();
  };
  $('#mlHorizon').oninput = onMlHorizonChange;
  $('#mlHorizon').onchange = onMlHorizonChange;
  $$('#spatialMetricSeg button').forEach(b => b.onclick = async () => {
    $$('#spatialMetricSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.mapMetric = b.dataset.map;
    await refreshSpatial();
  });
  $$('#heatModeSeg button').forEach(b => b.onclick = async () => {
    $$('#heatModeSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.heatMode = b.dataset.heat;
    await refreshHeat();
  });
  $$('#heatSortSeg button').forEach(b => b.onclick = async () => {
    $$('#heatSortSeg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.heatSort = b.dataset.sort;
    await refreshHeat();
  });
  $$('#view-overview .seg button').forEach(b => b.onclick = async () => {
    $$('#view-overview .seg button').forEach(x => x.classList.remove('active'));
    b.classList.add('active');
    state.mapMetric = b.dataset.map;
    state.renderToken += 1;
    await drawMap('map', state.mapMetric);
  });
  const onHorizonChange = async e => {
    state.forecastHorizon = clampInt(e.target.value, 1, 50, 3);
    e.target.value = state.forecastHorizon;
    if ($('#view-forecast').classList.contains('active-view')) await refreshForecast();
  };
  $('#forecastHorizon').oninput = onHorizonChange;
  $('#forecastHorizon').onchange = onHorizonChange;
  $('#closeModal').onclick = () => $('#stateModal').classList.add('hidden');
  window.addEventListener('resize', () => { clearTimeout(state._resizeTimer); state._resizeTimer = setTimeout(invalidateVisiblePlots, 80); });
}

function clampInt(v, min, max, fallback) {
  const n = Number.parseInt(v, 10);
  return Number.isFinite(n) ? Math.max(min, Math.min(max, n)) : fallback;
}

function switchView(v) {
  $$('.nav').forEach(b => b.classList.toggle('active', b.dataset.view === v));
  $$('.view').forEach(x => x.classList.remove('active-view'));
  $('#view-' + v).classList.add('active-view');

  const titles = {
    overview: ['OVERVIEW', 'National crime landscape'],
    spatial: ['SPATIAL', 'Where crime is concentrated'],
    temporal: ['TEMPORAL', 'How crime changes over time'],
    spatiotemporal: ['SPATIO-TEMPORAL', 'State × year patterns'],
    cri: ['CRI & RISK', 'Crime Risk Index and classification'],
    forecast: ['PREDICTION', 'Transparent growth-based forecast'],
    ml: ['ML MODELS', 'Compare deep learning vs statistical forecasting'],
    insights: ['INSIGHTS', 'Decision support and SDG 5'],
  };
  $('#viewEyebrow').textContent = titles[v][0];
  $('#viewTitle').textContent = titles[v][1];
  const isTemporal = v === 'temporal';
  const isForecast = v === 'forecast';
  const isMl = v === 'ml';
  const isSpatiotemporal = v === 'spatiotemporal';
  const stateRelevant = ['overview', 'temporal', 'forecast', 'ml'].includes(v);
  const yearRelevant = ['overview', 'spatial', 'cri', 'insights'].includes(v);
  const crimeRelevant = v !== 'cri' && v !== 'ml';
  $('#yearFilterGroup').classList.toggle('hidden', !yearRelevant);
  $('#forecastOriginGroup').classList.toggle('hidden', !isForecast);
  $('#temporalRange').classList.toggle('hidden', !isTemporal);
  $('#stateSelect').parentElement.classList.toggle('hidden', !stateRelevant);
  $('#crimeSelect').parentElement.classList.toggle('hidden', !crimeRelevant);
  $('#yearLabel').textContent = 'Year';
  if (isTemporal) {
    $('#startYearSelect').value = state.temporalStart;
    $('#endYearSelect').value = state.temporalEnd;
  }
  if (isForecast) {
    refreshForecastOptions()
      .then(() => refreshForecast())
      .catch(e => toast(e.message));
  } else if (isMl) {
    refreshMl().catch(e => toast(e.message));
  } else {
    refreshCurrent().catch(e => toast(e.message));
  }

  // Leaflet needs an explicit size recalculation after a hidden view becomes visible.
  setTimeout(() => {
    invalidateVisiblePlots();
    if (v === 'overview' && state.map) { try { state.map.invalidateSize(true); } catch (_) { } }
    if (v === 'spatial' && state.spatialMap) { try { state.spatialMap.invalidateSize(true); } catch (_) { } }
  }, 40);
}

function populateMeta(m) {
  $('#sourceBadge').textContent = m.source || 'No dataset';
  const ys = m.years || [];
  $('#yearSelect').innerHTML = ys.map(y => `<option value="${y}">${y}</option>`).join('');
  state.year = ys[ys.length - 1];
  $('#yearSelect').value = state.year;

  const current = state.state;
  const validState = current !== 'All' && m.states.includes(current) ? current : 'All';
  state.state = validState;
  $('#stateSelect').innerHTML = '<option value="All">All</option>' + m.states.map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join('');
  $('#stateSelect').value = state.state;

  $('#startYearSelect').innerHTML = ys.map(y => `<option value="${y}">${y}</option>`).join('');
  $('#endYearSelect').innerHTML = ys.map(y => `<option value="${y}">${y}</option>`).join('');
  state.temporalStart = state.temporalStart != null && ys.includes(state.temporalStart) ? state.temporalStart : ys[0];
  state.temporalEnd = state.temporalEnd != null && ys.includes(state.temporalEnd) ? state.temporalEnd : ys[ys.length - 1];
  if (state.temporalStart > state.temporalEnd) state.temporalStart = ys[0];
  state.forecastOrigin = state.forecastOrigin != null && ys.includes(state.forecastOrigin) ? state.forecastOrigin : ys[ys.length - 1];
  $('#startYearSelect').value = state.temporalStart;
  $('#endYearSelect').value = state.temporalEnd;
  $('#forecastOriginSelect').innerHTML = ys.map(y => `<option value="${y}">${y}</option>`).join('');
  $('#forecastOriginSelect').value = state.forecastOrigin;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}

async function refreshAll() {
  if (state.year == null) return;
  await refreshCurrent();
}

async function refreshOverview() {
  const token = ++state.renderToken;
  const o = await api(`/api/overview?year=${state.year}&crime=${encodeURIComponent(state.crime)}`);
  if (token !== state.renderToken) return;
  $('#kTotalLabel').textContent = state.crime === 'Total Crime' ? 'Total crime' : `${state.crime} cases`;
  $('#kTotal').textContent = fmt(o.selected_crime_total, 0);
  $('#kTotalMeta').textContent = `Year ${state.year}`;
  $('#kCri').textContent = fmt(o.avg_cri, 1);
  const catYearCaption = $('#categoryYearCaption');
  if (catYearCaption) catYearCaption.textContent = `Year ${state.year}`;
  $('#kHigh').textContent = o.high_risk;
  $('#kGrowth').textContent = `${o.avg_growth >= 0 ? '+' : ''}${fmt(o.avg_growth, 1)}%`;

  await drawMap('map', state.mapMetric);
  if (token !== state.renderToken) return;
  await drawRanking();
  if (token !== state.renderToken) return;
  await drawTrend();
  if (token !== state.renderToken) return;
  await drawCategories();
}

async function refreshCurrent() {
  const active = $('.active-view')?.id?.replace('view-', '');
  if (active === 'overview') return refreshOverview();
  if (active === 'spatial') return refreshSpatial();
  if (active === 'temporal') return refreshTemporal();
  if (active === 'spatiotemporal') return refreshHeat();
  if (active === 'cri') return refreshCri();
  if (active === 'forecast') return refreshForecast();
  if (active === 'ml') return refreshMl();
  if (active === 'insights') return refreshInsights();
}

async function loadGeo() {
  state.geo = null;
  for (const url of geoUrls) {
    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) continue;
      const geo = await r.json();
      if (geo && geo.type === 'FeatureCollection' && Array.isArray(geo.features) && geo.features.length) {
        state.geo = geo;
        return;
      }
    } catch (_) { }
  }
  toast('India boundary data is unavailable. Other analytics remain fully usable.');
}

function metricValue(d, metric) {
  if (metric === 'crime') return Number.isFinite(+d['Crime Value']) ? +d['Crime Value'] : 0;
  if (metric === 'growth') return Number.isFinite(+d['Growth Rate %']) ? +d['Growth Rate %'] : null;
  return Number.isFinite(+d.CRI) ? +d.CRI : 0;
}

function growthColorValue(rows = []) {
  const vals = rows.map(r => Number(r['Growth Rate %'])).filter(Number.isFinite);
  if (!vals.length) return 100;
  // Robust visual scale: extreme percentages caused by tiny prior-year bases
  // should not wash out the colors for all other states. Tooltip still shows
  // the exact percentage. 250% is the visual saturation point.
  const p90 = vals.map(Math.abs).sort((a, b) => a - b)[Math.floor((vals.length - 1) * 0.90)];
  return Math.max(50, Math.min(250, Number.isFinite(p90) ? p90 : 100));
}

function fillFor(v, metric, rows = []) {
  if (v == null || !Number.isFinite(+v)) return '#dfe5ef';
  if (metric === 'growth') {
    const cap = growthColorValue(rows);
    const x = Number(v);
    // Signed logarithmic transform gives useful separation for normal changes
    // while keeping extreme values from dominating the entire map.
    const scale = 25;
    const z = Math.asinh(Math.min(Math.abs(x), cap) / scale) / Math.asinh(cap / scale);
    const t = x >= 0 ? 0.5 + 0.5 * z : 0.5 - 0.5 * z;
    const neutral = '#f8fafc';
    const negative = '#dc2626';
    const positive = '#2563eb';
    if (t < 0.5) return lerpColor(negative, neutral, t * 2);
    return lerpColor(neutral, positive, (t - 0.5) * 2);
  }
  const maxValue = Math.max(1, ...rows.map(r => metricValue(r, metric)).filter(Number.isFinite));
  const t = Math.max(0, Math.min(1, Number(v) / maxValue));
  const stops = ['#eaf2ff', '#b7d0ff', '#6ea0ff', '#356fe6', '#173f9e'];
  const i = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
  const f = t * (stops.length - 1) - i;
  return lerpColor(stops[i], stops[i + 1], f);
}

function lerpColor(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ar = pa >> 16, ag = (pa >> 8) & 255, ab = pa & 255;
  const br = pb >> 16, bg = (pb >> 8) & 255, bb = pb & 255;
  return '#' + [ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t]
    .map(x => Math.round(x).toString(16).padStart(2, '0')).join('');
}

function removeGrowthLegend(id) {
  const key = id === 'map' ? 'map' : 'mapSpatial';
  const legend = state.growthLegends[key];
  if (legend) { try { legend.remove(); } catch (_) { } }
  state.growthLegends[key] = null;
}

function addGrowthLegend(map, id, rows) {
  removeGrowthLegend(id);
  const key = id === 'map' ? 'map' : 'mapSpatial';
  const cap = growthColorValue(rows);
  const LLegend = L.Control.extend({
    options: { position: 'bottomleft' },
    onAdd: function () {
      const div = L.DomUtil.create('div', 'growth-legend leaflet-control');
      div.innerHTML = `
        <div class="growth-legend-title">YoY growth</div>
        <div class="growth-legend-bar"></div>
        <div class="growth-legend-labels"><span>−${Math.round(cap)}%</span><span>0%</span><span>+${Math.round(cap)}%+</span></div>
      `;
      L.DomEvent.disableClickPropagation(div);
      return div;
    }
  });
  const legend = new LLegend();
  legend.addTo(map);
  state.growthLegends[key] = legend;
}

async function drawMap(id, metric) {
  const token = state.renderToken;
  const el = $('#' + id);
  if (!el) return;
  let map = id === 'map' ? state.map : state.spatialMap;
  if (!map) {
    map = L.map(id, { zoomControl: false, attributionControl: true, minZoom: 4, maxZoom: 7, scrollWheelZoom: false });
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    map.setView([22.5, 79], 4.5);
    if (id === 'map') state.map = map; else state.spatialMap = map;
  }
  if (!state.geo) {
    removeGrowthLegend(id);
    el.innerHTML = '<div class="map-unavailable">India state boundaries could not be loaded.<br>Try reloading with internet access enabled.</div>';
    return;
  }

  removeGrowthLegend(id);
  const data = await api(`/api/map?year=${state.year}&metric=${metric}&crime=${encodeURIComponent(state.crime)}`);
  if (token !== state.renderToken) return;
  const caption = metric === 'cri' ? 'CRI by state' : metric === 'crime' ? `${data.crime} cases by state` : `${data.crime} year-over-year growth by state`;
  if (id === 'map' && $('#mapCaption')) $('#mapCaption').textContent = caption;
  const by = Object.fromEntries(data.data.map(d => [d.state, d]));
  const layers = id === 'map' ? state.mapLayers : state.spatialLayers;
  layers.forEach(l => map.removeLayer(l));
  layers.length = 0;

  // Hidden/just-shown containers can report stale dimensions. Recalculate before drawing.
  map.invalidateSize(true);
  if (metric === 'growth') addGrowthLegend(map, id, data.data);
  const layer = L.geoJSON(state.geo, {
    style: f => {
      const n = normalizeGeo(f.properties);
      const d = resolveMapRecord(n, by);
      const v = d ? metricValue(d, metric) : null;
      return { fillColor: d ? fillFor(v, metric, data.data) : '#dfe5ef', weight: 0.8, color: '#fff', fillOpacity: 0.86 };
    },
    onEachFeature: (f, l) => {
      const n = normalizeGeo(f.properties);
      const d = resolveMapRecord(n, by);
      const crimeValue = d ? d['Crime Value'] : null;
      let tooltipHtml = `<b>${escapeHtml(n)}</b>`;
      if (d) {
        if (metric === 'cri') {
          tooltipHtml += `<br>CRI ${fmt(d.CRI)} • ${escapeHtml(d.Risk)}<br>Year ${data.year}`;
        } else if (metric === 'crime') {
          tooltipHtml += `<br>${escapeHtml(data.crime)} ${fmt(crimeValue, 0)}<br>Year ${data.year}`;
        } else {
          const gv = Number(d['Growth Rate %']);
          const gtxt = Number.isFinite(gv) ? `${gv >= 0 ? '+' : ''}${fmt(gv)}%` : '—';
          const warning = Number.isFinite(gv) && Math.abs(gv) >= 1000 ? '<br><small>Extreme % may reflect a very small prior-year base</small>' : '';
          tooltipHtml += `<br>${escapeHtml(data.crime)} growth <b>${gtxt}</b>${warning}<br>Year ${data.year}`;
        }
      } else {
        tooltipHtml += `<br>No data for selected year`;
      }
      l.bindTooltip(tooltipHtml, { sticky: true });
      l.on({
        mouseover: () => l.setStyle({ weight: 2, color: '#162033' }),
        mouseout: () => l.setStyle({ weight: 0.8, color: '#fff' }),
        click: () => d && by[n] && openState(n, metric),
      });
    },
  }).addTo(map);
  layers.push(layer);
  const bounds = layer.getBounds();
  if (bounds.isValid()) {
    map.invalidateSize(true);
    const pad = id === 'map' ? [44, 44] : [30, 30];
    map.fitBounds(bounds, { padding: pad, maxZoom: id === 'map' ? 4.2 : 4.8, animate: false });
  }
  setTimeout(() => { map.invalidateSize(true); if (bounds.isValid()) map.fitBounds(bounds, { padding: id === 'map' ? [44, 44] : [30, 30], maxZoom: id === 'map' ? 4.2 : 4.8, animate: false }); }, 80);
}

function normalizeGeo(p) {
  // India state GeoJSON sources use different property names.
  // The primary source used here exposes ST_NM, which must be checked explicitly.
  const raw = p?.ST_NM || p?.st_nm || p?.name || p?.NAME_1 || p?.STNAME ||
    p?.State_Name || p?.STATE_NAME || p?.state_name || p?.state || p?.NAME || '';
  const key = String(raw).trim().toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  const aliases = {
    'a n islands': 'Andaman and Nicobar Islands',
    'a n island': 'Andaman and Nicobar Islands',
    'andaman nicobar islands': 'Andaman and Nicobar Islands',
    'andaman and nicobar islands': 'Andaman and Nicobar Islands',
    'jammu kashmir': 'Jammu and Kashmir',
    'jammu and kashmir': 'Jammu and Kashmir',
    'jammu kashmir state': 'Jammu and Kashmir',
    'd n haveli': 'Dadra and Nagar Haveli',
    'd n haveli and daman diu': 'Dadra and Nagar Haveli',
    'dadra and nagar haveli': 'Dadra and Nagar Haveli',
    'daman diu': 'Daman and Diu',
    'daman and diu': 'Daman and Diu',
    'delhi ut': 'Delhi',
    'nct of delhi': 'Delhi',
    'orissa': 'Odisha',
    'pondicherry': 'Puducherry',
    'uttaranchal': 'Uttarakhand',
    'uttar pradesh': 'Uttar Pradesh',
    'tamilnadu': 'Tamil Nadu',
    'westbengal': 'West Bengal',
  };
  return aliases[key] || String(raw).trim();
}

function resolveMapRecord(name, by) {
  if (by[name]) return by[name];
  const key = String(name).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  // Contemporary India boundary files may merge the two UTs into one polygon.
  if (key === 'dadra and nagar haveli and daman and diu' || key === 'dadra nagar haveli and daman diu') {
    const a = by['Dadra and Nagar Haveli'];
    const b = by['Daman and Diu'];
    if (!a && !b) return null;
    const sum = k => (Number.isFinite(+a?.[k]) ? +a[k] : 0) + (Number.isFinite(+b?.[k]) ? +b[k] : 0);
    const growthVals = [a?.['Growth Rate %'], b?.['Growth Rate %']].map(Number).filter(Number.isFinite);
    const criVals = [a?.CRI, b?.CRI].map(Number).filter(Number.isFinite);
    const risk = [a?.Risk, b?.Risk].includes('High') ? 'High' : ([a?.Risk, b?.Risk].includes('Medium') ? 'Medium' : 'Low');
    return {
      state: 'Dadra and Nagar Haveli and Daman and Diu',
      'Crime Value': sum('Crime Value'),
      'Total Crime': sum('Total Crime'),
      CRI: criVals.length ? criVals.reduce((x, y) => x + y, 0) / criVals.length : null,
      Risk: risk,
      'Growth Rate %': growthVals.length ? growthVals.reduce((x, y) => x + y, 0) / growthVals.length : null,
    };
  }
  return null;
}

async function openState(s, metric = 'cri') {
  $('#stateSelect').value = s;
  state.state = s;
  $('#stateModal').classList.remove('hidden');
  $('#modalBody').innerHTML = '<p>Loading…</p>';
  try {
    const d = await api(`/api/state?state=${encodeURIComponent(s)}&year=${state.year}&crime=${encodeURIComponent(state.crime)}`);
    const hist = d.history || [];
    const selectedGrowth = hist.find(x => Number(x.year) === Number(d.year))?.['Growth Rate %'] ?? null;

    let summaryHtml = '';
    let chart;
    let title;
    let yTitle;

    if (metric === 'growth') {
      summaryHtml = `
        <div><span>Year-over-year growth</span><b class="growth-value ${selectedGrowth == null ? '' : (Number(selectedGrowth) >= 0 ? 'positive' : 'negative')}">${selectedGrowth == null ? '—' : (Number(selectedGrowth) >= 0 ? '+' : '') + fmt(selectedGrowth) + '%'}</b></div>
        <div><span>${escapeHtml(d.crime)}</span><b>${fmt(d.selected_value, 0)}</b></div>`;
      chart = {
        x: hist.map(x => x.year),
        y: hist.map(x => x['Growth Rate %']),
        mode: 'lines+markers',
        connectgaps: false,
        line: { width: 2, color: '#f97316' },
        marker: { size: 5 },
        name: 'Growth %',
        hovertemplate: '%{x}<br>Growth: %{y:.2f}%<extra></extra>'
      };
      title = `${d.crime} year-over-year growth`;
      yTitle = 'Growth %';
    } else if (metric === 'crime') {
      summaryHtml = `
        <div><span>${escapeHtml(d.crime)}</span><b>${fmt(d.selected_value, 0)}</b></div>`;
      chart = {
        x: hist.map(x => x.year),
        y: hist.map(x => x[d.crime]),
        mode: 'lines+markers',
        line: { width: 2 },
        marker: { size: 5 },
        name: d.crime,
        hovertemplate: '%{x}<br>${escapeHtml(d.crime)}: %{y:,.0f}<extra></extra>'
      };
      title = `${d.crime} trend`;
      yTitle = d.crime;
    } else {
      summaryHtml = `
        <div><span>CRI</span><b>${fmt(d.summary.CRI)}</b></div>
        <div><span>Risk</span><b>${escapeHtml(d.summary.Risk)}</b></div>`;
      // CRI is the selected spatial metric, so the modal shows only the state's CRI trend.
      const criSeries = d.cri_history || [];
      chart = {
        x: criSeries.map(x => x.year),
        y: criSeries.map(x => x.CRI),
        mode: 'lines+markers',
        line: { width: 2 },
        marker: { size: 5 },
        name: 'CRI',
        hovertemplate: '%{x}<br>CRI: %{y:.1f}<extra></extra>'
      };
      title = 'CRI trend';
      yTitle = 'CRI';
    }

    $('#modalBody').innerHTML = `
      <h3>${escapeHtml(s)}</h3>
      <div class="state-summary">${summaryHtml}</div>
      <div id="modalChart" class="modal-chart"></div>`;

    const layout = plotLayout(title);
    layout.yaxis = { ...(layout.yaxis || {}), title: yTitle, zeroline: metric === 'growth', zerolinecolor: '#cbd5e1', zerolinewidth: 1 };
    Plotly.newPlot('modalChart', [chart], layout, plotCfg);
  } catch (e) {
    $('#modalBody').innerHTML = `<p>${escapeHtml(e.message)}</p>`;
  }
}

async function drawRanking() {
  const d = await api(`/api/cri?year=${state.year}`);
  $('#ranking').innerHTML = d.data.map((r, i) => `
    <div class="rank-row" data-state="${escapeHtml(r.state)}">
      <span class="rank-no">${i + 1}</span>
      <span class="rank-name" title="${escapeHtml(r.state)}">${escapeHtml(r.state)}</span>
      <b>${fmt(r.CRI)}</b>
      <span class="risk ${r.Risk.toLowerCase()}">${r.Risk}</span>
    </div>`).join('');
  $$('.rank-row').forEach(r => r.onclick = () => openState(r.dataset.state, 'cri'));
}

async function drawTrend() {
  const d = await api(`/api/temporal?state=${encodeURIComponent(state.state)}&crime=${encodeURIComponent(state.crime)}`);
  $('#trendCaption').textContent = state.state === 'All' ? 'All states' : state.state;
  Plotly.newPlot('trendChart', [{
    x: d.data.map(x => x.year),
    y: d.data.map(x => x[state.crime]),
    mode: 'lines',
    fill: 'tozeroy',
    line: { width: 2 },
    name: state.crime,
  }], plotLayout('', true), plotCfg);
}

async function drawCategories() {
  const d = await api(`/api/categories?state=${encodeURIComponent(state.state)}&year=${state.year}`);
  const cats = Object.keys(d.categories);
  Plotly.newPlot('categoryChart', [{
    x: cats,
    y: cats.map(c => d.categories[c]),
    type: 'bar',
    marker: { color: '#2563eb' },
    hovertemplate: '%{x}: %{y:,}<extra></extra>',
  }], plotLayout('', true), plotCfg);
}

async function refreshSpatial() {
  const token = ++state.renderToken;
  $$('#spatialMetricSeg button').forEach(b => b.classList.toggle('active', b.dataset.map === state.mapMetric));
  await drawMap('mapSpatial', state.mapMetric);
  if (token !== state.renderToken) return;
  const d = await api(`/api/map?year=${state.year}&metric=${state.mapMetric}&crime=${encodeURIComponent(state.crime)}`);
  if (token !== state.renderToken) return;
  const rows = d.data.slice().sort((a, b) => metricValue(b, state.mapMetric) - metricValue(a, state.mapMetric)).slice(0, 12);
  const labelLen = Math.max(8, ...rows.map(r => String(r.state).length));
  const left = Math.min(240, Math.max(120, 22 + labelLen * 5.8));
  Plotly.newPlot('spatialBar', [{
    x: rows.map(r => metricValue(r, state.mapMetric)).reverse(),
    y: rows.map(r => r.state).reverse(),
    type: 'bar',
    orientation: 'h',
    marker: { color: '#2563eb' },
    hovertemplate: '%{y}: %{x}<extra></extra>',
  }], {
    ...plotLayout('', true),
    margin: { l: left, r: 12, t: 8, b: 28 },
    yaxis: { automargin: true, gridcolor: '#eef1f5', zeroline: false, tickfont: { size: 9 } },
    xaxis: { gridcolor: '#eef1f5', zeroline: false, tickfont: { size: 9 } },
  }, plotCfg);
}

async function refreshTemporal() {
  const token = ++state.renderToken;
  const start = state.temporalStart ?? state.year;
  const end = state.temporalEnd ?? state.year;
  const d = await api(`/api/temporal?state=${encodeURIComponent(state.state)}&crime=${encodeURIComponent(state.crime)}&start_year=${start}&end_year=${end}`);
  if (token !== state.renderToken) return;
  Plotly.newPlot('temporalChart', [
    {
      x: d.data.map(x => x.year),
      y: d.data.map(x => x[state.crime]),
      mode: 'lines+markers',
      line: { width: 2 },
      marker: { size: 5 },
      name: `${state.crime} cases`,
      hovertemplate: `<b>%{x}</b><br>${state.crime} cases: %{y:,}<extra></extra>`,
    },
    {
      x: d.data.map(x => x.year),
      y: d.data.map(x => x['Growth Rate %']),
      mode: 'lines+markers',
      yaxis: 'y2',
      line: { dash: 'dot', width: 1.7 },
      marker: { size: 4 },
      name: 'Year-over-year growth (%)',
      hovertemplate: `<b>%{x}</b><br>Year-over-year growth: %{y:.1f}%<extra></extra>`,
      connectgaps: false,
    },
  ], {
    ...plotLayout('', false),
    yaxis: { title: `${state.crime} cases`, automargin: true, zeroline: false },
    yaxis2: { title: 'Growth (%)', overlaying: 'y', side: 'right', showgrid: false, automargin: true, zeroline: false },
    legend: {
      orientation: 'h',
      x: 0,
      xanchor: 'left',
      y: 1.08,
      yanchor: 'bottom',
      font: { size: 9 },
      bgcolor: 'rgba(255,255,255,0.86)',
      bordercolor: '#e6eaf0',
      borderwidth: 1,
    },
  }, plotCfg);
  setTimeout(() => { const el = $('#temporalChart'); if (el && window.Plotly) { try { Plotly.Plots.resize(el); } catch (_) { } } }, 40);
  const vals = d.data.map(x => +x['Growth Rate %']).filter(Number.isFinite);
  const avg = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
  $('#growthInsight').textContent = `Average observed ${state.crime} growth from ${start}–${end} is ${avg >= 0 ? '+' : ''}${fmt(avg)}% per year. This is descriptive year-over-year change, not a causal effect.`;
  const rangeVals = d.data.map(x => +x[state.crime]).filter(Number.isFinite);
  const rangeAvg = rangeVals.length ? rangeVals.reduce((a, b) => a + b, 0) / rangeVals.length : 0;
  $('#avgInsight').textContent = `Across ${start}–${end}, the average annual ${state.crime} level is ${fmt(rangeAvg, 1)}. Use the Prediction page for forward estimates.`;
}

async function refreshHeat() {
  const token = ++state.renderToken;
  const d = await api(`/api/spatiotemporal?crime=${encodeURIComponent(state.crime)}&mode=volume&sort=severity`);
  if (token !== state.renderToken) return;

  const hoverText = d.raw_matrix.map((row, i) =>
    row.map((v, j) => v == null
      ? `<b>${d.states[i]} · ${d.years[j]}</b><br>❌ No data`
      : `<b>${d.states[i]} · ${d.years[j]}</b><br>${fmt(v, 0)} cases`
    )
  );

  const heatLeft = Math.min(220, Math.max(150, 20 + Math.max(...d.states.map(s => s.length)) * 5.6));
  const traces = [{
    z: d.raw_matrix, x: d.years, y: d.states,
    type: 'heatmap',
    colorscale: [[0, '#f0f9ff'], [0.25, '#bae6fd'], [0.5, '#38bdf8'], [0.75, '#0369a1'], [1, '#0c2461']],
    hoverinfo: 'text', text: hoverText,
    showscale: true, zsmooth: false,
    colorbar: { title: { text: 'Cases', font: { size: 10 } }, tickfont: { size: 9 }, thickness: 13, len: 0.9 },
  }];

  setTimeout(() => { try { Plotly.Plots.resize($('#heatmap')); } catch (_) { } }, 40);
  Plotly.newPlot('heatmap', traces, {
    ...plotLayout('', false),
    margin: { l: heatLeft, r: 20, t: 6, b: 28 },
    yaxis: { automargin: true, zeroline: false, tickfont: { size: 9 }, tickmode: 'array', tickvals: d.states, ticktext: d.states },
    xaxis: { zeroline: false, tickfont: { size: 9 }, tickmode: 'linear', dtick: 1 },
    hoverlabel: { font: { size: 11 } },
  }, plotCfg);
}




async function refreshCri() {
  const token = ++state.renderToken;
  const d = await api(`/api/cri?year=${state.year}`);
  if (token !== state.renderToken) return;
  const rows = d.data.slice().reverse();
  const criCaption = $('#criYearCaption');
  if (criCaption) criCaption.textContent = `Risk classification by state • ${state.year}`;
  const labelLen = Math.max(8, ...rows.map(r => String(r.state).length));
  const left = Math.min(250, Math.max(136, 24 + labelLen * 5.8));
  Plotly.newPlot('criChart', [{
    x: rows.map(r => r.CRI),
    y: rows.map(r => r.state),
    type: 'bar',
    orientation: 'h',
    marker: { color: rows.map(r => r.Risk === 'High' ? '#dc2626' : r.Risk === 'Medium' ? '#d97706' : '#16a34a') },
  }], {
    ...plotLayout('', true),
    margin: { l: left, r: 18, t: 8, b: 28 },
    yaxis: { automargin: true, gridcolor: '#eef1f5', zeroline: false, tickfont: { size: 9 } },
  }, plotCfg);
}

async function refreshForecastOptions() {
  const s = state.state || 'All';
  const d = await api(`/api/forecast-options?state=${encodeURIComponent(s)}`);
  const ys = d.years || [];
  const current = state.forecastOrigin;
  const target = current != null && ys.includes(current) ? current : ys[ys.length - 1];
  state.forecastOrigin = target;
  $('#forecastOriginSelect').innerHTML = ys.map(y => `<option value="${y}">${y}</option>`).join('');
  $('#forecastOriginSelect').value = target;
}

async function refreshForecast() {
  const token = ++state.forecastToken;
  const s = state.state || 'All';
  if (!$('#forecastOriginSelect').options.length) await refreshForecastOptions();
  const origin = Number($('#forecastOriginSelect').value || state.forecastOrigin);
  state.forecastOrigin = origin;
  const horizon = clampInt($('#forecastHorizon').value, 1, 50, 3);
  $('#forecastHorizon').value = horizon;
  const d = await api(`/api/forecast?state=${encodeURIComponent(s)}&crime=${encodeURIComponent(state.crime)}&origin_year=${origin}&horizon=${horizon}`);
  if (token !== state.forecastToken) return;

  $('#forecastTitle').textContent = `${d.scope_label || (s === 'All' ? 'All India' : s)} • ${state.crime}`;
  const endYear = d.forecast_end_year ?? (d.origin_year + horizon);
  const endValue = d.forecast_end_value;
  const change = d.projected_change_pct;
  $('#forecastSub').textContent = `Training cutoff: ${d.origin_year}. Forecast horizon: ${horizon} year${horizon === 1 ? '' : 's'} → ${endYear}. Growth rate is calculated only from observations through ${d.origin_year}.`;
  $('#fAvg').textContent = fmt(d.average, 1);
  $('#fRecent').textContent = fmt(d.recent_average, 1);
  $('#fGrowth').textContent = `${d.average_growth_rate >= 0 ? '+' : ''}${fmt(d.average_growth_rate)}%`;
  $('#fGrowthMeta').textContent = `${d.growth_observations ?? 0} YoY observations through ${d.origin_year} · damping ${fmt(d.damping_factor ?? 1, 2)}`;
  $('#fBase').textContent = fmt(d.forecast_baseline, 1);
  $('#fEnd').textContent = fmt(endValue, 1);
  $('#fEndMeta').textContent = change == null ? `by ${endYear}` : `${change >= 0 ? '+' : ''}${fmt(change, 1)}% by ${endYear}`;

  const historyX = d.history.map(x => x.year);
  const historyY = d.history.map(x => x[state.crime]);
  const forecastX = d.forecast.map(x => x.year);
  const forecastY = d.forecast.map(x => x.predicted);
  const actualFuture = d.forecast.map(x => x.actual);
  const hasActualFuture = actualFuture.some(v => v != null);

  const traces = [
    { x: historyX, y: historyY, mode: 'lines+markers', name: 'Historical', line: { width: 2 }, marker: { size: 4 } },
    { x: [d.origin_year, ...forecastX], y: [historyY[historyY.length - 1], ...forecastY], mode: 'lines+markers', name: 'Forecast', line: { dash: 'dash', width: 2 }, marker: { size: 5 } },
  ];
  if (hasActualFuture) {
    traces.push({ x: forecastX, y: actualFuture, mode: 'lines+markers', name: 'Actual future', line: { dash: 'dot', width: 1.5 }, marker: { size: 4 } });
  }

  traces[0].hovertemplate = `<b>%{x}</b><br>Historical: %{y:,.0f}<extra></extra>`;
  traces[1].hovertemplate = `<b>%{x}</b><br>Forecast: %{y:,.0f}<br>Projected growth: %{customdata:.2f}%<extra></extra>`;
  traces[1].customdata = [null, ...d.forecast.map(x => x.effective_growth_rate ?? null)];
  if (traces[2]) traces[2].hovertemplate = `<b>%{x}</b><br>Actual: %{y:,.0f}<extra></extra>`;

  const tickStep = (endYear - d.min_observed_year) <= 12 ? 1 : 2;
  const tickVals = [];
  for (let y = d.min_observed_year; y <= endYear; y += tickStep) tickVals.push(y);
  if (!tickVals.includes(endYear)) tickVals.push(endYear);

  Plotly.newPlot('forecastChart', traces, {
    ...plotLayout('', true),
    xaxis: {
      ...plotLayout('', true).xaxis,
      title: 'Year',
      range: [d.min_observed_year - 0.15, endYear + Math.max(0.5, horizon * 0.05)],
      tickmode: 'array',
      tickvals: tickVals,
      ticktext: tickVals.map(String),
      showgrid: true,
    },
    yaxis: { ...plotLayout('', true).yaxis, title: state.crime },
    showlegend: true,
    legend: { orientation: 'h', y: 1.08 },
  }, plotCfg);

  setTimeout(() => { const el = $('#forecastChart'); if (el && window.Plotly) { try { Plotly.Plots.resize(el); } catch (_) { } } }, 40);

  // Keep the QA metric visible without occupying chart area.
  const method = document.querySelector('.forecast-method span');
  if (method) {
    const maeText = d.backtest_mae == null ? 'Backtest MAE is unavailable for this early cutoff.' : `Rolling-origin MAE: ${fmt(d.backtest_mae, 1)}.`;
    const calibText = d.calibration_mae == null ? '' : ` Calibration MAE: ${fmt(d.calibration_mae, 1)}.`;
    method.textContent = `${d.method} ${maeText}${calibText}`;
  }
}

const ML_COLORS = { growth: '#8b5cf6', lstm: '#2563eb', ensemble: '#dc2626' };
const ML_LABELS = { growth: 'Damped Growth (Baseline)', lstm: 'LSTM (Deep Learning)', ensemble: 'Ensemble (Combined)' };

async function refreshMl() {
  const token = ++state.mlToken;
  const s = state.state || 'All';
  const isVal = state.mlMode === 'validation';
  const originYear = isVal ? 2018 : 2021;
  const horizon = isVal ? 3 : clampInt($('#mlHorizon').value, 1, 20, state.mlHorizon || 3);
  if (!isVal) $('#mlHorizon').value = horizon;
  const mode = state.mlModel;
  $('#mlCompareTable').classList.toggle('hidden', mode !== 'compare');

  const chartEl = $('#mlForecastChart');
  if (chartEl) chartEl.style.opacity = '0.5';
  $('#mlDescription').textContent = isVal
    ? 'Validating models against real held-out 2019–2021 data...'
    : 'Computing ML model predictions...';

  let d;
  try {
    if (mode === 'compare') {
      d = await api(`/api/forecast-compare-ml?state=${encodeURIComponent(s)}&horizon=${horizon}&origin_year=${originYear}`);
    } else {
      d = await api(`/api/forecast-ml?state=${encodeURIComponent(s)}&horizon=${horizon}&model=${mode}&origin_year=${originYear}`);
    }
  } catch (e) {
    if (chartEl) chartEl.style.opacity = '1';
    $('#mlDescription').textContent = e.message || 'ML models unavailable. Train and place them in models/ first.';
    Plotly.purge('mlForecastChart');
    return;
  }
  if (chartEl) chartEl.style.opacity = '1';
  if (token !== state.mlToken) return;

  $('#mlSub').textContent = `${d.scope_label || s} • Total Crime • trained through ${d.origin_year}`;

  const historyX = d.history.map(x => x.year);
  const historyY = d.history.map(x => x['Total Crime']);
  const originHistIdx = historyX.indexOf(d.origin_year);
  const originHistVal = originHistIdx >= 0 ? historyY[originHistIdx] : historyY[historyY.length - 1];

  const traces = [
    { x: historyX, y: historyY, mode: 'lines+markers', name: 'Historical Actual', line: { width: 2.5, color: '#162033' }, marker: { size: 5 } },
  ];

  if (mode === 'compare') {
    $('#mlDescription').textContent = isVal
      ? `Validation Mode (2019–2021 holdout): Comparing 3 models against actual recorded crime for ${d.scope_label} (black line).`
      : `Future Projection Mode: Comparing all 3 model trajectories on ${d.scope_label} for ${d.origin_year + 1}–${d.origin_year + horizon}.`;

    const tableHeader = isVal
      ? `<tr style="text-align:left;color:#718096"><th style="padding:4px">Model</th><th>${escapeHtml(d.scope_label)} Holdout MAE</th><th>Weighted Error (WAPE)</th><th>Forecast Bias</th><th>Trend Direction</th><th>Accuracy</th><th>Forecast end</th></tr>`
      : `<tr style="text-align:left;color:#718096"><th style="padding:4px">Model</th><th>Projected Volume (${d.origin_year + horizon})</th><th>90% Confidence Range</th><th>Expected Growth / yr</th></tr>`;

    const rows = ['<table style="width:100%;border-collapse:collapse;font-size:10px">', tableHeader];
    let bestModel = null;
    let minErr = Infinity;

    for (const key of ['growth', 'lstm', 'ensemble']) {
      const m = d.models[key];
      const fx = m.forecast.map(x => x.year);
      const fy = m.forecast.map(x => x.predicted);
      traces.push({
        x: [d.origin_year, ...fx], y: [originHistVal, ...fy],
        mode: 'lines+markers', name: ML_LABELS[key],
        line: { dash: 'dash', width: 2, color: ML_COLORS[key] }, marker: { size: 5, color: ML_COLORS[key] },
      });
      const bt = m.backtest || {};
      const sh = m.state_holdout;
      const fs = m.future_summary;

      if (isVal && sh) {
        if (sh.mae < minErr) { minErr = sh.mae; bestModel = ML_LABELS[key]; }
        const accBadge = `<span style="background:${sh.accuracy_pct >= 90 ? '#ecfdf5' : '#fef3c7'};color:${sh.accuracy_pct >= 90 ? '#047857' : '#92400e'};padding:1px 5px;border-radius:4px;font-weight:700">${fmt(sh.accuracy_pct, 1)}%</span>`;
        const dirBadge = `<span style="background:${sh.dir_accuracy_pct >= 67 ? '#eff6ff' : '#fef2f2'};color:${sh.dir_accuracy_pct >= 67 ? '#1d4ed8' : '#b91c1c'};padding:1px 5px;border-radius:4px;font-weight:600">${fmt(sh.dir_accuracy_pct, 0)}%</span>`;
        const biasBg = Math.abs(sh.bias_pct) <= 1.0 ? '#f3f4f6' : (sh.bias_pct > 1.0 ? '#fff1f2' : '#f0fdf4');
        const biasClr = Math.abs(sh.bias_pct) <= 1.0 ? '#374151' : (sh.bias_pct > 1.0 ? '#be123c' : '#15803d');
        const biasBadge = `<span style="background:${biasBg};color:${biasClr};padding:1px 5px;border-radius:4px;font-weight:600">${sh.bias_label}</span>`;

        rows.push(`<tr style="border-top:1px solid #f0f2f5"><td style="padding:4px;font-weight:700;color:${ML_COLORS[key]}">${ML_LABELS[key]}</td>` +
          `<td><b>${fmt(sh.mae, 0)}</b></td><td>${fmt(sh.wape_pct, 1)}%</td><td>${biasBadge}</td><td>${dirBadge}</td><td>${accBadge}</td>` +
          `<td>${fmt(fy[fy.length - 1], 0)} by ${fx[fx.length - 1]}</td></tr>`);
      } else {
        const growthText = fs ? `${fs.annual_growth_rate_pct >= 0 ? '+' : ''}${fmt(fs.annual_growth_rate_pct, 1)}% / yr` : '—';
        rows.push(`<tr style="border-top:1px solid #f0f2f5"><td style="padding:4px;font-weight:700;color:${ML_COLORS[key]}">${ML_LABELS[key]}</td>` +
          `<td><b>${fmt(fy[fy.length - 1], 0)}</b> cases</td><td>${fs?.confidence_range || '—'}</td><td>${growthText}</td></tr>`);
      }
    }
    rows.push('</table>');
    $('#mlCompareTable').innerHTML = rows.join('');

    if (isVal && bestModel) {
      if ($('#mlBtLbl1')) $('#mlBtLbl1').textContent = 'Holdout scope';
      if ($('#mlBtLbl2')) $('#mlBtLbl2').textContent = 'Top performing model';
      if ($('#mlBtLbl3')) $('#mlBtLbl3').textContent = 'Best Holdout MAE';
      if ($('#mlBtLbl4')) $('#mlBtLbl4').textContent = 'Validation period';
      $('#mlBtHorizon').textContent = `${d.scope_label}`;
      $('#mlBtMae').textContent = bestModel;
      $('#mlBtMapeMean').textContent = `${fmt(minErr, 0)} cases`;
      $('#mlBtMapeMedian').textContent = '2019–2021 (3 yrs)';
    } else {
      const fc = d.future_consensus;
      const ensPred = d.models?.ensemble?.forecast?.[horizon - 1]?.predicted;
      if ($('#mlBtLbl1')) $('#mlBtLbl1').textContent = 'Forecast Horizon';
      if ($('#mlBtLbl2')) $('#mlBtLbl2').textContent = `Ensemble Forecast (${d.origin_year + horizon})`;
      if ($('#mlBtLbl3')) $('#mlBtLbl3').textContent = 'Projection Range (Min – Max)';
      if ($('#mlBtLbl4')) $('#mlBtLbl4').textContent = 'Mean Across Models';

      $('#mlBtHorizon').textContent = `${d.origin_year + 1}–${d.origin_year + horizon} (h=${horizon})`;
      $('#mlBtMae').textContent = ensPred ? `${fmt(ensPred, 0)} cases` : '—';
      $('#mlBtMapeMean').textContent = fc ? `${fmt(fc.min_projected, 0)} to ${fmt(fc.max_projected, 0)}` : '—';
      $('#mlBtMapeMedian').textContent = fc ? `${fmt(fc.mean_projected, 0)} cases` : '—';
    }
  } else {
    $('#mlDescription').textContent = d.model_description || '—';
    const fx = d.forecast.map(x => x.year);
    const fy = d.forecast.map(x => x.predicted);

    if (!isVal && d.forecast[0]?.upper_bound != null) {
      const upperY = [originHistVal, ...d.forecast.map(x => x.upper_bound)];
      const lowerY = [originHistVal, ...d.forecast.map(x => x.lower_bound)];
      const bandX = [d.origin_year, ...fx];
      const bandBg = mode === 'ensemble' ? 'rgba(220, 38, 38, 0.12)' : (mode === 'lstm' ? 'rgba(37, 99, 235, 0.12)' : 'rgba(139, 92, 246, 0.12)');
      traces.unshift({
        x: [...bandX, ...bandX.slice().reverse()],
        y: [...upperY, ...lowerY.slice().reverse()],
        fill: 'toself',
        fillcolor: bandBg,
        line: { color: 'transparent' },
        name: '90% Confidence Interval',
        showlegend: true,
        hoverinfo: 'skip',
      });
    }

    traces.push({
      x: [d.origin_year, ...fx], y: [originHistVal, ...fy],
      mode: 'lines+markers', name: d.model_label,
      line: { dash: 'dash', width: 2, color: ML_COLORS[mode] }, marker: { size: 5, color: ML_COLORS[mode] },
    });
    const bt = d.backtest || {};
    const sh = d.state_holdout;
    const fs = d.future_summary;

    if (isVal && sh) {
      if ($('#mlBtLbl1')) $('#mlBtLbl1').textContent = 'Validation Scope';
      if ($('#mlBtLbl2')) $('#mlBtLbl2').textContent = `${d.scope_label} Holdout MAE`;
      if ($('#mlBtLbl3')) $('#mlBtLbl3').textContent = 'Weighted Error (WAPE)';
      if ($('#mlBtLbl4')) $('#mlBtLbl4').textContent = 'Forecast Bias';

      $('#mlBtHorizon').textContent = `${d.scope_label} (2019–2021)`;
      $('#mlBtMae').textContent = `${fmt(sh.mae, 0)} cases`;
      $('#mlBtMapeMean').textContent = `${fmt(sh.wape_pct, 1)}%`;
      $('#mlBtMapeMedian').textContent = sh.bias_label;
    } else {
      if ($('#mlBtLbl1')) $('#mlBtLbl1').textContent = 'Forecast Horizon';
      if ($('#mlBtLbl2')) $('#mlBtLbl2').textContent = `Projected Volume (${d.origin_year + horizon})`;
      if ($('#mlBtLbl3')) $('#mlBtLbl3').textContent = 'Expected Annual Growth';
      if ($('#mlBtLbl4')) $('#mlBtLbl4').textContent = '90% Confidence Range';

      $('#mlBtHorizon').textContent = `${d.origin_year + 1}–${d.origin_year + horizon} (h=${horizon})`;
      $('#mlBtMae').textContent = fs ? `${fmt(fs.projected_crime, 0)} cases` : '—';
      $('#mlBtMapeMean').textContent = fs ? `${fs.annual_growth_rate_pct >= 0 ? '+' : ''}${fmt(fs.annual_growth_rate_pct, 1)}% / yr` : '—';
      $('#mlBtMapeMedian').textContent = fs ? `±${fmt(fs.margin_of_error_pct, 1)}% (${fs.confidence_range})` : '—';
    }
  }

  const endYear = d.origin_year + horizon;
  Plotly.newPlot('mlForecastChart', traces, {
    ...plotLayout('', true),
    xaxis: { ...plotLayout('', true).xaxis, title: 'Year', showgrid: true },
    yaxis: { ...plotLayout('', true).yaxis, title: 'Total Crime' },
    showlegend: true,
    legend: { orientation: 'h', y: 1.1 },
  }, plotCfg);
  setTimeout(() => { const el = $('#mlForecastChart'); if (el && window.Plotly) { try { Plotly.Plots.resize(el); } catch (_) { } } }, 40);
}

async function refreshInsights() {
  const token = ++state.renderToken;
  const c = await api(`/api/cri?year=${state.year}`);
  if (token !== state.renderToken) return;
  const top = c.data.slice(0, 5);
  const g = await api(`/api/map?year=${state.year}&metric=growth&crime=${encodeURIComponent(state.crime)}`);
  if (token !== state.renderToken) return;
  const fast = g.data.filter(x => Number.isFinite(+x['Growth Rate %'])).slice().sort((a, b) => (+b['Growth Rate %']) - (+a['Growth Rate %'])).slice(0, 5);
  const crime = g.data.slice().sort((a, b) => b['Crime Value'] - a['Crime Value']).slice(0, 5);
  $('#riskInsights').innerHTML = list(top.map(x => `${x.state}: CRI ${fmt(x.CRI)} (${x.Risk})`));
  $('#temporalInsights').innerHTML = list(fast.map(x => `${x.state}: ${x['Growth Rate %'] >= 0 ? '+' : ''}${fmt(x['Growth Rate %'])}% ${state.crime} growth`));
  $('#categoryInsights').innerHTML = list(crime.map(x => `${x.state}: ${fmt(x['Crime Value'], 0)} ${state.crime} cases`));
}

function list(a) {
  return `<ul class="insight-list">${a.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul>`;
}

function plotLayout(title = '', compact = false) {
  return {
    margin: { l: compact ? 34 : 50, r: compact ? 18 : 45, t: title ? 34 : 8, b: 28 },
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui', size: 9, color: '#718096' },
    xaxis: { gridcolor: '#eef1f5', zeroline: false },
    yaxis: { gridcolor: '#eef1f5', zeroline: false },
    showlegend: false,
    title,
  };
}

init();
