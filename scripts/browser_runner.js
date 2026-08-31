async function getPageWsUrl() {
  const res = await fetch('http://localhost:9222/json/list');
  const targets = await res.json();
  const page = targets.find((t) => t.type === 'page' && t.url.includes('5173'));
  if (!page) {
    throw new Error('Could not find page target on port 5173');
  }
  return page.webSocketDebuggerUrl;
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.ws = null;
    this.id = 0;
    this.callbacks = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const { resolve, reject } = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) reject(msg.error);
          else resolve(msg.result);
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const msgId = ++this.id;
      this.callbacks.set(msgId, { resolve, reject });
      this.ws.send(JSON.stringify({ id: msgId, method, params }));
    });
  }

  async eval(expression) {
    const res = await this.send('Runtime.evaluate', {
      expression,
      returnByValue: true,
      awaitPromise: true,
    });
    if (res.exceptionDetails) {
      throw new Error(JSON.stringify(res.exceptionDetails));
    }
    return res.result?.value;
  }
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function run() {
  console.log('🚀 Connecting to Chromium DevTools on port 9222...');
  const wsUrl = await getPageWsUrl();
  const cdp = new CDPClient(wsUrl);
  await cdp.connect();
  console.log('✅ Connected to browser tab!');

  await cdp.send('Page.enable');
  await cdp.send('Runtime.enable');

  // Reload page to ensure latest JS assets are active
  console.log('🔄 Reloading page on http://localhost:5173...');
  await cdp.send('Page.reload');
  await sleep(1500);

  // 1. Switch to Training View
  console.log('\n--- 1. Navigating to Training Mode (Key 2) ---');
  await cdp.eval(`
    (() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '2' }));
    })()
  `);
  await sleep(1000);

  const algorithms = [
    { id: 'ppo', name: 'CleanRL PPO' },
    { id: 'alphazero', name: 'AlphaZero Dual Head' },
    { id: 'recurrent_ppo', name: 'Recurrent PPO (LSTM)' },
    { id: 'self_play_ppo', name: 'Self-Play Policy Pool' },
    { id: 'dqn', name: 'Double-DQN' },
  ];

  // 2. Train each of the 5 algorithms at 1,000,000 steps
  for (let i = 0; i < algorithms.length; i++) {
    const algo = algorithms[i];
    console.log(`\n======================================================`);
    console.log(`▶ [${i + 1}/5] Browser UI Training: ${algo.name} (${algo.id}) at 1,000,000 steps`);
    console.log(`======================================================`);

    // Click algorithm card
    console.log(`👉 Clicking algorithm card: ${algo.name}...`);
    const selected = await cdp.eval(`
      (() => {
        const cards = Array.from(document.querySelectorAll('.training-view div[style*="cursor: pointer"]'));
        const card = cards.find(c => c.textContent.includes('${algo.name}') || c.textContent.includes('${algo.id}'));
        if (card) {
          card.click();
          return true;
        }
        return false;
      })()
    `);
    console.log(`Card selection result: ${selected}`);
    await sleep(600);

    // Select 1,000,000 steps in dropdown
    console.log(`⚙️ Selecting timesteps: 1,000,000 steps (1M) in dropdown...`);
    const val = await cdp.eval(`
      (() => {
        const selects = Array.from(document.querySelectorAll('.training-view select'));
        const tsSelect = selects.find(s => Array.from(s.options).some(o => o.value === '1000000' || o.value === '500000'));
        if (tsSelect) {
          const opt1M = Array.from(tsSelect.options).find(o => o.value === '1000000');
          if (opt1M) {
            tsSelect.value = '1000000';
          } else {
            tsSelect.selectedIndex = tsSelect.options.length - 1;
          }
          tsSelect.dispatchEvent(new Event('change', { bubbles: true }));
          return tsSelect.value;
        }
        return null;
      })()
    `);
    console.log(`Selected timesteps value: ${val}`);
    await sleep(600);

    // Click Ignite Training button
    console.log(`🔥 Clicking 'Ignite ${algo.name} Training' button...`);
    const igniteClicked = await cdp.eval(`
      (() => {
        const buttons = Array.from(document.querySelectorAll('.training-view button'));
        const igniteBtn = buttons.find(b => b.textContent.toLowerCase().includes('ignite'));
        if (igniteBtn) {
          igniteBtn.click();
          return true;
        }
        return false;
      })()
    `);
    console.log(`Ignite button clicked: ${igniteClicked}`);
    await sleep(1500);

    // Monitor training progress
    console.log(`⏳ Monitoring live training progress on page...`);
    let done = false;
    let pollCount = 0;
    while (!done) {
      await sleep(1000);
      pollCount++;
      const status = await cdp.eval(`
        (() => {
          const isTrainingBtn = Array.from(document.querySelectorAll('button')).some(b => b.textContent.includes('Abort Training'));
          const hasIgnite = Array.from(document.querySelectorAll('button')).some(b => b.textContent.includes('Ignite'));
          const progressCard = Array.from(document.querySelectorAll('.telemetry-card, div')).find(d => d.textContent && d.textContent.includes('Training Progress'));
          const textContent = progressCard ? progressCard.innerText.split('\\n').join(' | ') : '';
          return {
            isTraining: isTrainingBtn,
            hasIgnite,
            sampleText: textContent,
          };
        })()
      `);

      if (pollCount % 5 === 0 || !status?.isTraining) {
        console.log(`   [${pollCount}s] isTraining: ${status?.isTraining} | Progress: ${status?.sampleText}`);
      }

      if (!status?.isTraining && status?.hasIgnite && pollCount > 1) {
        done = true;
        console.log(`✅ Training finished for ${algo.name}!`);
      }

      if (pollCount > 600) {
        console.log(`⚠️ Reached safety poll limit (600s), proceeding...`);
        break;
      }
    }

    await sleep(1000);
  }

  // 3. Switch to Tournament View
  console.log('\n--- 3. Navigating to Tournament Arena (Key 4) ---');
  await cdp.eval(`
    (() => {
      window.dispatchEvent(new KeyboardEvent('keydown', { key: '4' }));
    })()
  `);
  await sleep(1000);

  // 4. Open Contestant Selection Panel
  console.log('⚙️ Opening Contestants Selection drawer...');
  await cdp.eval(`
    (() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const partBtn = btns.find(b => b.textContent.includes('Participants'));
      if (partBtn) partBtn.click();
    })()
  `);
  await sleep(800);

  // Set 30 matches per pair (10x games)
  console.log('⚙️ Setting matches per pair to 30 (10x increase)...');
  await cdp.eval(`
    (() => {
      const selects = Array.from(document.querySelectorAll('.elo-matrix-heatmap select'));
      const gamesSelect = selects.find(s => Array.from(s.options).some(o => o.value === '30' || o.value === '10'));
      if (gamesSelect) {
        const opt30 = Array.from(gamesSelect.options).find(o => o.value === '30');
        if (opt30) {
          gamesSelect.value = '30';
        } else {
          gamesSelect.selectedIndex = gamesSelect.options.length - 1;
        }
        gamesSelect.dispatchEvent(new Event('change', { bubbles: true }));
        return gamesSelect.value;
      }
      return null;
    })()
  `);
  await sleep(600);

  // Select all participants (5 models + 3 baselines)
  console.log('📋 Selecting participants in UI...');
  const selectionInfo = await cdp.eval(`
    (() => {
      const allBtn = Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('All ('));
      if (allBtn) {
        allBtn.click();
      }
      const selected = Array.from(document.querySelectorAll('.elo-matrix-heatmap div[style*="cursor: pointer"]'))
        .map(el => el.innerText.split('\\n')[0])
        .filter(Boolean);
      return selected;
    })()
  `);
  console.log('Selected contestants in UI:', selectionInfo);
  await sleep(600);

  // 5. Click Run Tournament
  console.log('\n🏆 Clicking "Run Tournament" / "Launch Simulation"...');
  const tourneyStarted = await cdp.eval(`
    (() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const runBtn = btns.find(b => b.textContent.includes('Run Tournament') || b.textContent.includes('Launch Simulation'));
      if (runBtn) {
        runBtn.click();
        return true;
      }
      return false;
    })()
  `);
  console.log(`Tournament launched from browser: ${tourneyStarted}`);

  // 6. Monitor tournament completion
  let tourneyDone = false;
  let tourneyPoll = 0;
  while (!tourneyDone) {
    await sleep(1000);
    tourneyPoll++;
    const state = await cdp.eval(`
      (() => {
        const isSimulating = Array.from(document.querySelectorAll('button')).some(b => b.textContent.includes('Simulating...'));
        const rows = Array.from(document.querySelectorAll('table tbody tr')).map(r => {
          return Array.from(r.querySelectorAll('td')).map(td => td.innerText.trim());
        }).filter(r => r.length > 3);

        return {
          isSimulating,
          rowsCount: rows.length,
          leaderboard: rows,
        };
      })()
    `);

    if (tourneyPoll % 5 === 0 || !state?.isSimulating) {
      console.log(`   [Tournament ${tourneyPoll}s] Simulating: ${state?.isSimulating}, Contestants ranked: ${state?.rowsCount}`);
    }

    if (!state?.isSimulating && state?.rowsCount >= 5 && tourneyPoll > 3) {
      tourneyDone = true;
      console.log('\n🎉 TOURNAMENT COMPLETED SUCCESSFULLY!');
      console.log('======================================================');
      console.log('🏆 FINAL TOURNAMENT LEADERBOARD (from Browser UI):');
      console.log('======================================================');
      console.table(state.leaderboard.map(r => ({
        Rank_Name: r[0],
        Elo: r[1],
        WinRate: r[2],
        W_L_D: r[3],
        AvgScore: r[4],
      })));
      break;
    }

    if (tourneyPoll > 600) {
      console.log('⚠️ Reached tournament timeout.');
      break;
    }
  }

  console.log('\n✅ All browser actions finished!');
  process.exit(0);
}

run().catch((err) => {
  console.error('❌ Error during browser automation:', err);
  process.exit(1);
});
