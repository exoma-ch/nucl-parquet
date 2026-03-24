<script>
  let {
    label, min, max, step = 0.1,
    low = $bindable(), high = $bindable(),
    formatValue = (v) => v.toFixed(1),
    parseValue = null,  // display string → internal value; if null, numbers edit the raw internal value
    unit = '',          // string or function(v) → string
    oninput = () => {},
  } = $props();

  // Display values for the number inputs (real-scale, not log)
  let displayLow = $derived(formatValue(low));
  let displayHigh = $derived(formatValue(high));
  let unitLow = $derived(typeof unit === 'function' ? unit(low) : unit);
  let unitHigh = $derived(typeof unit === 'function' ? unit(high) : unit);

  function parseInput(text) {
    if (parseValue) return parseValue(text);
    const v = parseFloat(text);
    return isNaN(v) ? null : v;
  }

  function clampLow(e) {
    const v = parseInput(e.target.value);
    if (v != null) low = Math.max(min, Math.min(v, high));
    oninput();
  }
  function clampHigh(e) {
    const v = parseInput(e.target.value);
    if (v != null) high = Math.min(max, Math.max(v, low));
    oninput();
  }
  function onSliderLow(e) {
    const v = parseFloat(e.target.value);
    low = Math.min(v, high);
    oninput();
  }
  function onSliderHigh(e) {
    const v = parseFloat(e.target.value);
    high = Math.max(v, low);
    oninput();
  }

  let pctLow = $derived(((low - min) / (max - min)) * 100);
  let pctHigh = $derived(((high - min) / (max - min)) * 100);
</script>

<div class="range-input">
  <label class="range-label">{label}</label>
  <div class="range-row">
    <div class="range-num-wrap">
      <input type="text" class="range-num" value={displayLow} onchange={clampLow} />
      {#if unitLow}<span class="range-unit">{unitLow}</span>{/if}
    </div>
    <div class="range-track-wrapper">
      <div class="range-track">
        <div class="range-fill" style="left: {pctLow}%; right: {100 - pctHigh}%;"></div>
      </div>
      <input type="range" class="range-thumb thumb-low" {min} {max} {step} value={low} oninput={onSliderLow} />
      <input type="range" class="range-thumb thumb-high" {min} {max} {step} value={high} oninput={onSliderHigh} />
    </div>
    <div class="range-num-wrap">
      <input type="text" class="range-num" value={displayHigh} onchange={clampHigh} />
      {#if unitHigh}<span class="range-unit">{unitHigh}</span>{/if}
    </div>
  </div>
</div>

<style>
  .range-input { display: flex; flex-direction: column; gap: 4px; }
  .range-label {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--text-dim); font-weight: 600;
  }
  .range-row { display: flex; align-items: center; gap: 6px; }
  .range-num-wrap {
    display: flex; align-items: center; gap: 2px; flex-shrink: 0;
  }
  .range-num {
    width: 52px; font-size: 11px; font-family: var(--mono);
    background: var(--bg3); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); padding: 3px 4px; text-align: right;
  }
  .range-num:focus { outline: 1px solid var(--accent); }
  .range-unit {
    font-size: 10px; color: var(--text-dim); white-space: nowrap;
  }
  .range-track-wrapper {
    flex: 1; position: relative; height: 24px;
  }
  .range-track {
    position: absolute; top: 50%; left: 0; right: 0;
    height: 4px; transform: translateY(-50%);
    background: var(--border); border-radius: 2px;
  }
  .range-fill {
    position: absolute; top: 0; bottom: 0;
    background: var(--accent); border-radius: 2px; opacity: 0.5;
  }
  .range-thumb {
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
    -webkit-appearance: none; appearance: none; background: none;
    pointer-events: none; margin: 0;
  }
  .range-thumb::-webkit-slider-thumb {
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); border: 2px solid var(--bg);
    cursor: pointer; pointer-events: auto;
  }
  .range-thumb::-moz-range-thumb {
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--accent); border: 2px solid var(--bg);
    cursor: pointer; pointer-events: auto;
  }
</style>
