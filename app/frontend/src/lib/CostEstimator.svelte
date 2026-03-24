<script>
  let { z, a } = $props();

  let data = $state(null);
  let loading = $state(false);

  const FACTOR_LABELS = {
    target_abundance: 'Target abundance',
    peak_cross_section: 'Peak cross-section',
    route_type: 'Route type',
    half_life: 'Half-life (shipping)',
    route_flexibility: 'Route flexibility',
  };
  const FACTOR_COLORS = {
    target_abundance: '#ffa726',
    peak_cross_section: '#6d9eeb',
    route_type: '#93c47d',
    half_life: '#ef5350',
    route_flexibility: '#ab47bc',
  };

  async function load() {
    loading = true;
    const res = await fetch(`/api/cost-estimate/${z}/${a}`);
    data = await res.json();
    loading = false;
  }

  $effect(() => {
    void z; void a;
    if (z && a) load();
  });

  function scoreColor(s) {
    if (s < 30) return '#ef5350';
    if (s < 60) return '#ffa726';
    return '#93c47d';
  }
</script>

<div class="cost-estimator">
  {#if loading}
    <p class="loading">Estimating cost...</p>
  {:else if data}
    <div class="cost-header">
      <div class="cost-score" style="color: {scoreColor(data.score)};">
        {data.score.toFixed(0)}<span class="cost-unit">/100</span>
      </div>
      <div class="cost-label">
        Production feasibility
        <span class="cost-hint">(higher = easier/cheaper)</span>
      </div>
    </div>
    <div class="cost-factors">
      {#each Object.entries(data.factors) as [key, value]}
        <div class="factor-row">
          <span class="factor-name">{FACTOR_LABELS[key] || key}</span>
          <div class="factor-bar-track">
            <div
              class="factor-bar-fill"
              style="width: {value}%; background: {FACTOR_COLORS[key] || '#6d9eeb'};"
            ></div>
          </div>
          <span class="factor-val">{value.toFixed(0)}</span>
        </div>
      {/each}
    </div>
    {#if data.notes}
      <p class="cost-notes">{data.notes}</p>
    {/if}
  {/if}
</div>

<style>
  .cost-estimator { margin-top: 8px; }
  .cost-header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
  .cost-score {
    font-size: 28px; font-weight: 800; font-family: var(--mono);
  }
  .cost-unit { font-size: 14px; font-weight: 400; color: var(--text-dim); }
  .cost-label { font-size: 13px; color: var(--text); }
  .cost-hint { font-size: 11px; color: var(--text-dim); }
  .cost-factors { display: flex; flex-direction: column; gap: 5px; }
  .factor-row { display: flex; align-items: center; gap: 8px; }
  .factor-name {
    width: 130px; font-size: 11px; color: var(--text-dim);
    text-align: right; flex-shrink: 0;
  }
  .factor-bar-track {
    flex: 1; height: 10px; background: var(--bg);
    border-radius: 3px; overflow: hidden;
  }
  .factor-bar-fill { height: 100%; border-radius: 3px; transition: width 0.3s; }
  .factor-val { width: 28px; font-size: 11px; font-family: var(--mono); color: var(--text); }
  .cost-notes {
    margin-top: 8px; font-size: 11px; color: var(--text-dim); font-style: italic;
  }
</style>
