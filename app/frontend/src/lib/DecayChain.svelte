<script>
  let { z, a } = $props();

  let chain = $state(null);
  let loading = $state(false);

  const TYPE_COLORS = {
    auger: '#ffa726', ce: '#93c47d', 'beta-': '#6d9eeb',
    alpha: '#ef5350', gamma: '#ab47bc', xray: '#78909c',
  };

  async function load() {
    loading = true;
    const res = await fetch(`/api/decay-chain/${z}/${a}?max_gen=10`);
    chain = await res.json();
    loading = false;
  }

  $effect(() => {
    void z; void a;
    if (z && a) load();
  });

  function maxDose(nodes) {
    let m = 0;
    for (const n of nodes) {
      const t = Object.values(n.dose_by_type).reduce((a, b) => a + b, 0);
      if (t > m) m = t;
    }
    return m || 1;
  }

  function eqLabel(eq) {
    if (eq === 'secular') return 'Secular eq.';
    if (eq === 'transient') return 'Transient eq.';
    return '';
  }

  function hlLabel(s) {
    if (!s || isNaN(s)) return 'stable';
    if (s < 60) return `${s.toFixed(1)}s`;
    if (s < 3600) return `${(s / 60).toFixed(1)}m`;
    if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
    if (s < 86400 * 365.25) return `${(s / 86400).toFixed(1)}d`;
    return `${(s / (86400 * 365.25)).toFixed(1)}y`;
  }
</script>

<div class="decay-chain">
  {#if loading}
    <p class="loading">Loading decay chain...</p>
  {:else if chain && chain.nodes?.length}
    {@const md = maxDose(chain.nodes)}
    <div class="chain-waterfall">
      {#each chain.nodes as node, i}
        <div class="chain-node" class:first={i === 0}>
          <div class="node-header">
            <span class="node-isotope">{node.symbol}-{node.A}</span>
            <span class="node-hl">{hlLabel(node.half_life_s)}</span>
            {#if node.decay_mode}
              <span class="node-decay">{node.decay_mode}</span>
            {/if}
            {#if node.equilibrium}
              <span class="node-eq">{eqLabel(node.equilibrium)}</span>
            {/if}
            {#if node.cum_branching != null}
              <span class="node-br">BR: {(node.cum_branching * 100).toFixed(1)}%</span>
            {/if}
          </div>
          <!-- Stacked dose bar -->
          <div class="dose-bar-track">
            {#each Object.entries(node.dose_by_type) as [type, dose]}
              {#if dose > 0}
                <div
                  class="dose-bar-seg"
                  style="width: {(dose / md) * 100}%; background: {TYPE_COLORS[type] || '#555'};"
                  title="{type}: {dose.toFixed(5)} MeV/Bq·s"
                ></div>
              {/if}
            {/each}
          </div>
          {#if i < chain.nodes.length - 1}
            <div class="chain-arrow">&#x2193;</div>
          {/if}
        </div>
      {/each}
    </div>
    {#if chain.cumulative_dose}
      <div class="chain-summary">
        Total chain dose: <strong>{chain.cumulative_dose.toFixed(4)}</strong> MeV/Bq·s
      </div>
    {/if}
  {:else if chain}
    <p style="color: var(--text-dim); font-size: 12px;">No decay chain data (stable isotope?)</p>
  {/if}
</div>

<style>
  .decay-chain { margin-top: 8px; }
  .chain-waterfall { display: flex; flex-direction: column; gap: 2px; }
  .chain-node { padding: 4px 0; }
  .node-header {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; margin-bottom: 3px; flex-wrap: wrap;
  }
  .node-isotope { font-weight: 700; color: #e8eaed; font-family: var(--mono); }
  .node-hl { color: var(--text-dim); font-family: var(--mono); font-size: 11px; }
  .node-decay {
    background: rgba(109,158,235,0.15); color: var(--accent);
    padding: 1px 5px; border-radius: 3px; font-size: 10px; font-family: var(--mono);
  }
  .node-eq {
    font-size: 10px; color: #93c47d; font-style: italic;
  }
  .node-br {
    font-size: 10px; color: var(--text-dim); font-family: var(--mono);
  }
  .dose-bar-track {
    display: flex; height: 14px; border-radius: 3px;
    overflow: hidden; background: var(--bg);
  }
  .dose-bar-seg { height: 100%; min-width: 1px; }
  .chain-arrow {
    text-align: center; color: var(--text-dim); font-size: 12px;
    line-height: 1; padding: 1px 0;
  }
  .chain-summary {
    margin-top: 8px; font-size: 12px; color: var(--text);
    font-family: var(--mono); padding: 6px 8px;
    background: var(--bg); border-radius: 4px;
  }
</style>
