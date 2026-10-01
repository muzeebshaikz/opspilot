"""Single-file UI: sign-in, chat, tickets, incidents, evaluation, system status.

Still not the deliverable frontend -- spec §24 calls for Next.js + TypeScript +
Tailwind and that scaffold lives in `frontend/`. This page exists so the whole
backend contract is demonstrable without a Node toolchain: the SSE event stream,
the three distinct outcomes (answered / abstained / denied), the approval pause
and resume, citation rendering, and the live token budget.

The sign-in screen is deliberately honest about what it is: a passwordless
account picker that mints a real signed session token. The session mechanism is
real; the identity proof is not.
"""

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OpsPilot AI</title>
<style>
:root{
  --bg:#0e1015; --panel:#151922; --panel2:#1b202b; --line:#262c3a; --fg:#e8eaf0;
  --mut:#8a93a8; --accent:#5b8cff; --accent2:#7aa2ff; --ok:#3ecf8e; --warn:#f5b23a;
  --danger:#ff6b6b; --radius:10px;
}
@media (prefers-color-scheme: light){
  :root:not([data-theme="dark"]){
    --bg:#f6f7fa; --panel:#ffffff; --panel2:#f0f2f7; --line:#dfe3ec; --fg:#141824;
    --mut:#5b6478; --accent:#2f62e0;
  }
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  background:var(--bg);color:var(--fg);display:flex;overflow:hidden}
button{font:inherit;cursor:pointer;border:0;border-radius:8px;padding:9px 16px;
  background:var(--accent);color:#fff;font-weight:500}
button:hover{background:var(--accent2)}
button.ghost{background:var(--panel2);color:var(--mut);border:1px solid var(--line)}
button:disabled{opacity:.5;cursor:default}
input,select{font:inherit;background:var(--panel2);border:1px solid var(--line);
  border-radius:8px;padding:10px 12px;color:var(--fg);width:100%}

/* ---------- login ----------
   The overlay scrolls and the card is a flex column with a scrollable account
   list, so the Sign in button is ALWAYS visible. A fixed grid with centred
   content clips the button on short viewports, which reads as "nothing
   happened when I clicked". */
#login{position:fixed;inset:0;background:var(--bg);z-index:50;
  display:flex;align-items:center;justify-content:center;
  padding:24px;overflow-y:auto}
.card{width:440px;max-width:92vw;max-height:calc(100vh - 48px);
  background:var(--panel);border:1px solid var(--line);
  border-radius:14px;padding:0 24px 0;overflow-y:auto;position:relative}
.card h1{margin:0;padding:22px 0 4px;font-size:21px;display:flex;align-items:center;gap:9px}
.card p.sub{margin:0 0 12px;color:var(--mut);font-size:13.5px}
/* Sticky footer: the button stays reachable no matter how short the window is,
   and every account stays scrollable to. */
.signbar{position:sticky;bottom:0;background:var(--panel);
  padding:10px 0 20px;margin-top:4px}
#loginError{color:var(--danger);font-size:13px;min-height:0;margin-bottom:6px}
.note{margin-top:0 !important}
.acct{display:flex;align-items:center;gap:11px;padding:11px;border:1px solid var(--line);
  border-radius:9px;margin-bottom:8px;cursor:pointer;background:var(--panel2)}
.acct:hover{border-color:var(--accent)}
.acct.sel{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 12%,transparent)}
.acct .ini{width:34px;height:34px;border-radius:9px;background:var(--accent);color:#fff;
  display:grid;place-items:center;font-weight:600;font-size:13px;flex:0 0 auto}
.acct .nm{font-weight:500}
.acct .meta{font-size:12px;color:var(--mut)}
.acct.off{opacity:.55}
.note{margin-top:12px;font-size:11.5px;color:var(--mut);border-top:1px solid var(--line);
  padding-top:10px;line-height:1.45;flex:0 0 auto}

/* ---------- shell ---------- */
aside{width:228px;background:var(--panel);border-right:1px solid var(--line);
  display:flex;flex-direction:column;padding:16px 12px;flex:0 0 auto}
aside h2{font-size:15px;margin:0 0 16px;display:flex;align-items:center;gap:8px;padding:0 6px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--ok)}
.nav{display:flex;flex-direction:column;gap:2px}
.nav a{color:var(--mut);text-decoration:none;padding:8px 10px;border-radius:8px;font-size:13.5px}
.nav a:hover{background:var(--panel2);color:var(--fg)}
.nav a.on{background:var(--panel2);color:var(--fg);font-weight:500}
.who{margin-top:auto;border-top:1px solid var(--line);padding-top:12px;font-size:12px;color:var(--mut)}
.who .nm{color:var(--fg);font-weight:500;font-size:13px}
main{flex:1;display:flex;flex-direction:column;min-width:0}
header{padding:13px 22px;border-bottom:1px solid var(--line);display:flex;
  justify-content:space-between;align-items:center;gap:14px}
header b{font-size:15px}
.budget{font-size:11.5px;color:var(--mut);display:flex;align-items:center;gap:7px}
.bar{width:90px;height:5px;border-radius:3px;background:var(--line);overflow:hidden}
.bar i{display:block;height:100%;background:var(--ok)}
.view{flex:1;overflow-y:auto;padding:22px;display:none}
.view.on{display:block}
#chatview.on{display:flex;flex-direction:column;padding:0}

/* ---------- chat ---------- */
#log{flex:1;overflow-y:auto;padding:22px;display:flex;flex-direction:column;gap:16px}
.msg{max-width:860px;display:flex;gap:11px}
.msg .av{width:28px;height:28px;border-radius:8px;flex:0 0 auto;display:grid;
  place-items:center;font-size:11px;font-weight:600;color:#fff}
.user .av{background:#3b4356}.bot .av{background:var(--accent)}
.bubble{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  padding:13px 15px;min-width:0;white-space:pre-wrap;word-wrap:break-word}
.user .bubble{background:var(--panel2)}
.activity{font-size:12.5px;color:var(--mut);border-left:2px solid var(--line);
  padding-left:10px;margin-bottom:10px;display:none}
.activity.show{display:block}
.cites{margin-top:12px;border-top:1px solid var(--line);padding-top:10px;font-size:12.5px}
.cites b{color:var(--mut);font-weight:500;display:block;margin-bottom:6px}
.cite{color:var(--mut);padding:3px 0}
.cite i{color:var(--accent);font-style:normal;font-weight:600}
.badges{margin-top:11px;display:flex;gap:6px;flex-wrap:wrap}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line);color:var(--mut)}
.badge.ok{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
.badge.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
.badge.danger{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,transparent)}
.outcome{margin:-2px 0 10px;font-size:12.5px;font-weight:500;display:flex;align-items:center;gap:7px}
.outcome.abstain{color:var(--warn)}.outcome.denied{color:var(--danger)}.outcome.ok{color:var(--ok)}
.approve{margin-top:13px;background:color-mix(in srgb,var(--accent) 10%,transparent);
  border:1px solid color-mix(in srgb,var(--accent) 35%,transparent);border-radius:9px;padding:13px}
.approve p{margin:0 0 11px;font-size:13.5px}
.newchat{width:100%;margin:0 0 10px;background:var(--panel2);color:var(--fg);
  border:1px solid var(--line);font-size:13px;padding:7px}
.newchat:hover{border-color:var(--accent)}
.convs{margin-top:14px;border-top:1px solid var(--line);padding-top:10px;
  overflow-y:auto;max-height:26vh}
.convhead{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--mut);padding:0 8px 6px}
.convitem{font-size:12.5px;color:var(--mut);padding:5px 8px;border-radius:6px;
  cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.convitem:hover{background:var(--panel2);color:var(--fg)}
.convitem.on{background:var(--panel2);color:var(--fg)}
#tone{width:auto;flex:0 0 auto;font-size:13px;padding:8px 10px}
.acts{display:flex;gap:5px;margin-top:9px;align-items:center}
.iconbtn{background:none;border:1px solid var(--line);color:var(--mut);
  border-radius:6px;padding:3px 9px;font-size:12px;font-weight:400}
.iconbtn:hover{border-color:var(--accent);color:var(--fg);background:var(--panel2)}
.iconbtn.sel{border-color:var(--ok);color:var(--ok)}
.iconbtn.selbad{border-color:var(--danger);color:var(--danger)}
.fbstatus{font-size:11px}
.memtag{font-size:11px;color:var(--mut);border:1px dashed var(--line);
  border-radius:20px;padding:2px 8px;margin-left:auto}
tr.clickable{cursor:pointer}
tr.clickable:hover td{background:var(--panel2)}
.detail{margin-top:14px;background:var(--panel);border:1px solid var(--line);
  border-radius:10px;padding:14px;font-size:13.5px}
.detail h4{margin:0 0 8px;font-size:14px}
.detail dl{display:grid;grid-template-columns:150px 1fr;gap:5px 12px;margin:0}
.detail dt{color:var(--mut)}
.chips{display:flex;gap:7px;flex-wrap:wrap;padding:0 22px 12px}
.chip{font-size:12px;border:1px solid var(--line);border-radius:20px;padding:6px 12px;
  color:var(--mut);cursor:pointer;background:var(--panel)}
.chip:hover{border-color:var(--accent);color:var(--fg)}
footer{border-top:1px solid var(--line);padding:13px 22px;display:flex;gap:10px}
footer input{flex:1}

/* ---------- tables ---------- */
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line)}
th{color:var(--mut);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:hover td{background:var(--panel2)}
.pill{font-size:11px;padding:2px 8px;border-radius:20px;border:1px solid var(--line)}
.pill.active,.pill.open{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 40%,transparent)}
.pill.resolved,.pill.closed{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 40%,transparent)}
.pill.Critical,.pill.SEV1{color:var(--danger);border-color:color-mix(in srgb,var(--danger) 40%,transparent)}
h3{margin:0 0 4px;font-size:16px}
.hint{color:var(--mut);font-size:13px;margin:0 0 16px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:20px}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:14px}
.stat .k{font-size:11.5px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-size:22px;font-weight:600;margin-top:5px}
.stat .v small{font-size:12px;color:var(--mut);font-weight:400}
pre{background:var(--panel);border:1px solid var(--line);padding:13px;border-radius:9px;
  overflow:auto;font-size:12px;line-height:1.5}
.empty{color:var(--mut);padding:26px;text-align:center;border:1px dashed var(--line);border-radius:9px}
</style>
</head>
<body>

<!-- ============ LOGIN ============ -->
<div id="login">
  <div class="card">
    <h1><span class="dot"></span> OpsPilot AI</h1>
    <p class="sub">Sign in to the Acme Technologies IT copilot.</p>
    <p class="sub" style="margin:-8px 0 12px">Click an account to sign in.</p>
    <div id="accounts"></div>
    <div class="signbar">
      <div id="loginError"></div>
      <button id="signin" style="width:100%" disabled>Sign in</button>
    </div>
    <div class="note" style="padding-bottom:20px">
      <b>Demo authentication.</b> Selecting an account mints a real signed,
      expiring session token that the server verifies on every request — but
      there is no password and no identity provider, so this proves the
      <i>session</i> mechanism, not identity. Authorization (what each role may
      do) is fully enforced server-side.
    </div>
  </div>
</div>

<!-- ============ SHELL ============ -->
<aside>
  <h2><span class="dot"></span> OpsPilot AI</h2>
  <button class="newchat" onclick="newConv()">+ New chat</button>
  <div class="nav">
    <a href="#" data-v="chat" class="on">Chat</a>
    <a href="#" data-v="tickets">My tickets</a>
    <a href="#" data-v="incidents">Active incidents</a>
    <a href="#" data-v="evaluation" data-staff="1">Evaluation</a>
    <a href="#" data-v="system" data-staff="1">System status</a>
  </div>
  <div class="convs"><div class="convhead">Recent</div><div id="convlist"></div></div>
  <div class="who">
    <div class="nm" id="whoName">—</div>
    <div id="whoRole">—</div>
    <button class="ghost" style="width:100%;margin-top:10px;padding:6px" onclick="logout()">Sign out</button>
  </div>
</aside>

<main>
  <header>
    <b id="title">Chat</b>
    <div class="budget" title="Groq free tier: 8,000 tokens/minute">
      <span id="budgetTxt">budget —</span><span class="bar"><i id="budgetBar" style="width:0%"></i></span>
    </div>
  </header>

  <div id="chatview" class="view on">
    <div id="log"></div>
    <div class="chips">
      <div class="chip" onclick="ask(this.textContent)">What does ERR_VPN_809 mean and how do I fix it?</div>
      <div class="chip" onclick="ask(this.textContent)">Is there currently a VPN outage?</div>
      <div class="chip" onclick="ask(this.textContent)">My VPN keeps disconnecting. Check for an outage and check my laptop.</div>
      <div class="chip" onclick="ask(this.textContent)">What is the status of INC-10482?</div>
      <div class="chip" onclick="ask(this.textContent)">Create a high priority ticket for this VPN issue</div>
      <div class="chip" onclick="ask(this.textContent)">Show me every employee's device information</div>
      <div class="chip" onclick="ask(this.textContent)">What is the procedure for tape backup retrieval?</div>
    </div>
    <footer>
      <select id="tone" title="Changes how the answer reads, never what it is grounded in">
        <option value="standard">Standard</option>
        <option value="concise">Concise</option>
        <option value="steps">Step-by-step</option>
        <option value="detailed">Detailed</option>
      </select>
      <input id="q" placeholder="Ask about IT support, incidents or your tickets…"
             onkeydown="if(event.key==='Enter')send()"/>
      <button id="send" onclick="send()">Send</button>
      <button id="stop" onclick="stopGen()" style="display:none;background:var(--danger)">Stop</button>
    </footer>
  </div>

  <div id="ticketsview" class="view"></div>
  <div id="incidentsview" class="view"></div>
  <div id="evaluationview" class="view"></div>
  <div id="systemview" class="view"></div>
</main>

<script>
let TOKEN = localStorage.getItem('opspilot_token') || null;
let turnSeq = 0, lastCites = [];
let controller = null;   // aborts the in-flight stream
let userStopped = false; // distinguishes a deliberate Stop from a real failure
let ME = null, conv = null, selected = null;
const $ = s => document.querySelector(s);
const esc = s => (s??'').toString().replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const H = () => TOKEN ? {'Authorization':'Bearer '+TOKEN} : {};

/* ---------- auth ---------- */
async function loadAccounts(){
  const r = await fetch('/api/v1/auth/directory');
  const d = await r.json();
  $('#accounts').innerHTML = d.accounts.map(a => `
    <div class="acct ${a.status==='offboarded'?'off':''}" data-id="${a.employee_id}">
      <div class="ini">${esc(a.name.split(' ').map(x=>x[0]).join('').slice(0,2))}</div>
      <div style="flex:1">
        <div class="nm">${esc(a.name)}</div>
        <div class="meta">${esc(a.role)} · ${esc(a.department)}${a.status!=='active'?' · '+esc(a.status):''}</div>
      </div>
    </div>`).join('');
  // One click signs in. Requiring select-then-confirm looked like the card was
  // simply not responding, because the only feedback was a thin border change.
  document.querySelectorAll('.acct').forEach(el => el.onclick = () => {
    document.querySelectorAll('.acct').forEach(x=>x.classList.remove('sel'));
    el.classList.add('sel');
    selected = el.dataset.id;
    $('#signin').disabled = false;
    signIn();
  });
}

async function signIn(){
  if(!selected) return;
  const btn = $('#signin');
  btn.disabled = true;
  btn.textContent = 'Signing in…';
  $('#loginError').textContent = '';
  try{
    const r = await fetch('/api/v1/auth/login', {method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({employee_id:selected})});
    const d = await r.json();
    if(!r.ok){
      // Offboarded accounts are REJECTED on purpose -- that is the account-state
      // check working, not a broken button. Say so instead of failing silently.
      $('#loginError').textContent = d.detail || 'Sign-in failed.';
      btn.disabled = false; btn.textContent = 'Sign in';
      return;
    }
    TOKEN = d.token;
    localStorage.setItem('opspilot_token', TOKEN);
    await boot();
  }catch(e){
    $('#loginError').textContent = 'Could not reach the server. Is it running on this port?';
    btn.disabled = false; btn.textContent = 'Sign in';
  }
}
$('#signin').onclick = signIn;
function logout(){ TOKEN=null; localStorage.removeItem('opspilot_token'); location.reload(); }

async function boot(){
  const r = await fetch('/api/v1/auth/me', {headers:H()});
  if(!r.ok){ TOKEN=null; localStorage.removeItem('opspilot_token'); $('#login').style.display='grid'; loadAccounts(); return; }
  ME = await r.json();
  $('#login').style.display='none';
  $('#whoName').textContent = ME.name;
  $('#whoRole').textContent = ME.role + ' · ' + ME.department;
  // Evaluation and System status span the whole estate and are operator views,
  // not end-user views. Hidden for employees; the server enforces this too.
  document.querySelectorAll('.nav a[data-staff]').forEach(a => {
    a.style.display = ME.is_staff ? '' : 'none';
  });
  loadConvs();
  refreshBudget(); setInterval(refreshBudget, 15000);
}

/* ---------- nav ---------- */
function showView(v){
  document.querySelectorAll('.nav a').forEach(x=>x.classList.remove('on'));
  const link = document.querySelector(`.nav a[data-v="${v}"]`);
  if(link){ link.classList.add('on'); $('#title').textContent = link.textContent; }
  document.querySelectorAll('.view').forEach(x=>x.classList.remove('on'));
  $('#'+v+'view').classList.add('on');
  if(v!=='chat') render(v);
}
document.querySelectorAll('.nav a[data-v]').forEach(a => a.onclick = e => {
  e.preventDefault();
  showView(a.dataset.v);
});

/* ---------- budget ---------- */
async function refreshBudget(){
  try{
    const m = await (await fetch('/metrics')).json();
    const b = m.token_budget || {};
    if(b.tpm_limit){
      const pct = Math.min(100, 100*b.tokens_this_minute/b.tpm_limit);
      $('#budgetTxt').textContent = `${b.tokens_this_minute}/${b.tpm_limit} tok/min`;
      $('#budgetBar').style.width = pct+'%';
      $('#budgetBar').style.background = pct>80 ? 'var(--danger)' : pct>50 ? 'var(--warn)' : 'var(--ok)';
    } else { $('#budgetTxt').textContent = 'offline provider'; }
  }catch(e){}
}

/* ---------- chat ---------- */
function add(role, html){
  const d = document.createElement('div');
  d.className = 'msg '+role;
  d.innerHTML = `<div class="av">${role==='user'?'You':'AI'}</div><div class="bubble">${html}</div>`;
  $('#log').appendChild(d); $('#log').scrollTop = 1e9;
  return d.querySelector('.bubble');
}
function ask(t){ $('#q').value = t; send(); }
function askAbout(t){ showView('chat'); ask(t); }

function stopGen(){
  // Aborts the HTTP stream. The server sees the disconnect and stops sending.
  // A model call already in flight still finishes and is still billed — the
  // banner says so rather than implying the cost was avoided.
  // No-op when nothing is streaming, so a late click cannot fake a failure.
  if(!controller) return;
  userStopped = true;
  controller.abort();
  controller = null;
}

function newConv(){
  stopGen();
  conv = null;
  $('#log').innerHTML = '';
  document.querySelectorAll('.convitem').forEach(x => x.classList.remove('on'));
  showView('chat');
}

async function loadConvs(){
  try{
    const d = await (await fetch('/api/v1/conversations', {headers:H()})).json();
    const items = (d.conversations || []).slice(0, 12);
    $('#convlist').innerHTML = items.length
      ? items.map(c => `<div class="convitem ${c.conversation_id===conv?'on':''}"
             data-id="${c.conversation_id}" title="${esc(c.title)}">${esc(c.title)}</div>`).join('')
      : '<div class="convitem" style="cursor:default">No conversations yet</div>';
    document.querySelectorAll('.convitem[data-id]').forEach(el =>
      el.onclick = () => openConv(el.dataset.id));
  }catch(e){ /* the sidebar is advisory; never break the page for it */ }
}

async function openConv(id){
  stopGen();
  try{
    const d = await (await fetch('/api/v1/conversations/'+id, {headers:H()})).json();
    conv = id;
    $('#log').innerHTML = '';
    showView('chat');
    (d.messages || []).forEach(m => {
      const b = add(m.role === 'user' ? 'user' : 'bot', esc(m.content));
      if(m.citations?.length){
        b.innerHTML += '<div class="cites"><b>Sources</b>' + m.citations.map(c =>
          `<div class="cite"><i>[${c.index}]</i> ${esc(c.title)}</div>`).join('') + '</div>';
      }
    });
    loadConvs();
  }catch(e){}
}

const OUTCOMES = {
  answered:            {cls:'ok',      icon:'&#10003;', text:'Answered from the knowledge base'},
  abstained_no_evidence:{cls:'abstain',icon:'&#9888;',  text:'Abstained — not enough evidence to answer safely'},
  partial_evidence:    {cls:'abstain',icon:'&#8776;',  text:'Related material only — not a confirmed answer'},
  denied_authorization:{cls:'denied',  icon:'&#128274;',text:'Denied — your role does not permit this'},
  clarification:       {cls:'',        icon:'&#63;',    text:'Needs clarification'}
};

async function send(){
  if(!TOKEN) return;
  const inp = $('#q'), text = inp.value.trim();
  if(!text) return;
  inp.value='';
  $('#send').style.display='none'; $('#stop').style.display='';
  userStopped = false;
  controller = new AbortController();
  add('user', esc(text));
  const b = add('bot', '<div class="activity"></div><div class="outcome" style="display:none"></div><div class="ans"></div>');
  const act = b.querySelector('.activity'), out = b.querySelector('.outcome'), ans = b.querySelector('.ans');

  let res;
  try{
    res = await fetch('/api/v1/chat/stream', {method:'POST',
      headers:{'Content-Type':'application/json', ...H()},
      body: JSON.stringify({message:text, conversation_id:conv,
                            tone: $('#tone').value}),
      signal: controller.signal});
  }catch(e){
    ans.textContent = e.name === 'AbortError' ? '' : 'Network error.';
    finishTurn(); return;
  }

  if(res.status === 429 || res.status === 503){
    const d = await res.json().catch(()=>({}));
    out.style.display='flex'; out.className='outcome abstain';
    out.innerHTML = '&#9888; ' + esc(d.message || 'Rate limited — retry shortly.');
    $('#send').disabled=false; refreshBudget(); return;
  }

  const rd = res.body.getReader(), dec = new TextDecoder();
  let buf='', stopped = false, completed = false;
  try{
  while(true){
    const {done, value} = await rd.read(); if(done) break;
    buf += dec.decode(value, {stream:true});
    const frames = buf.split('\n\n'); buf = frames.pop();
    for(const f of frames){
      const ev = (f.match(/^event: (.*)$/m)||[])[1];
      const dm = f.match(/^data: (.*)$/m); if(!ev||!dm) continue;
      const d = JSON.parse(dm[1]);
      if(ev==='start') conv = d.conversation_id;
      else if(ev==='activity'){ act.classList.add('show'); act.innerHTML += '&#10003; '+esc(d.step)+'<br>'; }
      else if(ev==='token') ans.textContent += d.text;
      else if(ev==='approval_required'){
        ans.innerHTML += `<div class="approve"><p><b>Approval needed.</b><br>${esc(d.description)}</p>
          <button onclick="approve('${d.token}',this)">Approve</button>
          <button class="ghost" onclick="decline('${d.token}',this)">Not now</button></div>`;
      }
      else if(ev==='citations'){
        lastCites = d.citations || [];
        if(!d.citations?.length) continue;
        ans.innerHTML += '<div class="cites"><b>Sources</b>' + d.citations.map(c =>
          `<div class="cite"><i>[${c.index}]</i> ${esc(c.title)} — ${esc(c.section)} · v${esc(c.version)} · ${esc(c.status)}</div>`).join('') + '</div>';
      }
      else if(ev==='done'){
        completed = true;
        const o = OUTCOMES[d.outcome] || OUTCOMES.answered;
        out.style.display='flex'; out.className='outcome '+o.cls;
        out.innerHTML = o.icon+' '+o.text;
        const bs=[`<span class="badge">${d.latency_ms} ms</span>`];
        if(d.usage?.llm_calls!==undefined) bs.push(`<span class="badge">${d.usage.llm_calls} LLM call(s)</span>`);
        if(d.usage?.cached) bs.push('<span class="badge ok">cache hit</span>');
        if(d.usage?.prompt_tokens) bs.push(`<span class="badge">${d.usage.prompt_tokens} prompt tok</span>`);
        ans.innerHTML += `<div class="badges">${bs.join('')}</div>`;

        // Action bar. Feedback is stored with the system's OWN confidence, so a
        // thumbs-down on a confident answer becomes a regression candidate.
        const turnId = 'turn'+(++turnSeq);
        const payload = encodeURIComponent(JSON.stringify({
          conversation_id: conv, question: text,
          answer: ans.innerText.slice(0,4000), outcome: d.outcome,
          citations: (lastCites||[]).length}));
        const memTag = d.memory_used
          ? '<span class="memtag" title="Earlier turns in this conversation were used as context">using conversation context</span>'
          : '';
        ans.innerHTML += `<div class="acts" id="${turnId}" data-rating="">
          <button class="iconbtn rate" data-r="up" onclick="rate('${turnId}','up','${payload}')" title="Helpful">&#128077;</button>
          <button class="iconbtn rate" data-r="down" onclick="rate('${turnId}','down','${payload}')" title="Not helpful">&#128078;</button>
          <button class="iconbtn" onclick="copyAns(this)" title="Copy answer">Copy</button>
          <button class="iconbtn" onclick="ask(${JSON.stringify(text).replace(/"/g,'&quot;')})" title="Ask again">Regenerate</button>
          <span class="fbstatus"></span>
          ${memTag}</div>`;
        refreshBudget(); loadConvs();
      }
      $('#log').scrollTop = 1e9;
    }
  }
  }catch(e){
    if(userStopped || e.name === 'AbortError'){ stopped = true; }
    else { ans.innerHTML += '<div class="badges"><span class="badge danger">stream error</span></div>'; }
  }
  // A Stop pressed after the answer already landed is not an interruption.
  if(stopped && completed) stopped = false;
  if(stopped){
    out.style.display='flex'; out.className='outcome abstain';
    out.innerHTML = '&#9632; Stopped by you';
    ans.innerHTML += `<div class="badges">
      <span class="badge warn">stopped</span>
      <span class="badge" title="Python cannot kill the worker thread, so a model call already in flight finishes and is still billed">model call already issued still counts against quota</span>
    </div>`;
    refreshBudget();
  }
  finishTurn();
}

function finishTurn(){
  controller = null;
  $('#send').style.display=''; $('#stop').style.display='none';
  $('#send').disabled = false;
}

async function rate(turnId, rating, payload){
  const bar = document.getElementById(turnId);
  if(!bar || bar.dataset.busy === '1') return;      // ignore double-clicks
  const status = bar.querySelector('.fbstatus');
  const current = bar.dataset.rating || '';
  // Clicking the same thumb again clears the rating rather than stacking.
  const next = current === rating ? '' : rating;

  const body = JSON.parse(decodeURIComponent(payload));
  body.rating = next || rating;
  body.retracted = next === '';

  bar.dataset.busy = '1';
  status.textContent = 'saving…';
  status.className = 'fbstatus badge';
  try{
    const r = await fetch('/api/v1/feedback', {method:'POST',
      headers:{'Content-Type':'application/json', ...H()}, body: JSON.stringify(body)});
    if(!r.ok) throw new Error('failed');
    bar.dataset.rating = next;
    // Exactly one status element, replaced in place -- never appended.
    bar.querySelectorAll('.iconbtn.rate').forEach(b=>b.classList.remove('sel','selbad'));
    if(next){
      bar.querySelector(`.iconbtn.rate[data-r="${next}"]`)
         .classList.add(next==='up'?'sel':'selbad');
      status.textContent = 'thanks, recorded';
      status.className = 'fbstatus badge' + (next==='up'?' ok':'');
    } else {
      status.textContent = '';
      status.className = 'fbstatus';
    }
  }catch(e){
    status.textContent = 'could not record';
    status.className = 'fbstatus badge danger';
  }finally{
    bar.dataset.busy = '0';
  }
}

function copyAns(btn){
  const t = btn.closest('.ans')?.innerText || '';
  navigator.clipboard.writeText(t).then(()=>{
    const old = btn.textContent; btn.textContent = 'Copied';
    setTimeout(()=>btn.textContent = old, 1400);
  });
}

async function approve(tok, btn){
  btn.disabled=true; btn.textContent='Working…';
  const r = await fetch('/api/v1/approvals/'+tok, {method:'POST', headers:H()});
  const d = await r.json();
  btn.parentElement.innerHTML = r.ok
    ? `<span class="badge ok">${esc(d.status)}</span> ${esc(d.answer||'')}`
    : `<span class="badge danger">${esc(d.detail||'failed')}</span>`;
}
async function decline(tok, btn){
  await fetch('/api/v1/approvals/'+tok+'/decline', {method:'POST', headers:H()});
  btn.parentElement.innerHTML = '<span class="badge">not now</span>';
}

/* ---------- panels ---------- */
async function render(v){
  const el = $('#'+v+'view'); el.innerHTML = '<div class="empty">Loading…</div>';
  try{
    if(v==='tickets'){
      const d = await (await fetch('/api/v1/tickets', {headers:H()})).json();
      const t = d.tickets||[];
      el.innerHTML = `<h3>My tickets</h3><p class="hint">Scoped server-side to your own records — an employee cannot read another employee's ticket.</p>`
        + (t.length ? `<table><tr><th>ID</th><th>Summary</th><th>Category</th><th>Priority</th><th>Status</th><th>Team</th></tr>`
          + t.map((x,i)=>`<tr class="clickable" data-i="${i}"><td>${esc(x.ticket_id)}</td><td>${esc(x.summary)}</td><td>${esc(x.category)}</td>
             <td><span class="pill ${esc(x.priority)}">${esc(x.priority)}</span></td>
             <td><span class="pill ${esc(x.status)}">${esc(x.status)}</span></td><td>${esc(x.assigned_team)}</td></tr>`).join('')
          + `</table><div id="ticketDetail"></div>` : '<div class="empty">No tickets.</div>');
      el.querySelectorAll('tr.clickable').forEach(tr => tr.onclick = () => {
        const x = t[+tr.dataset.i];
        $('#ticketDetail').innerHTML = `<div class="detail">
          <h4>${esc(x.ticket_id)} — ${esc(x.summary)}</h4>
          <dl><dt>Status</dt><dd>${esc(x.status)}</dd>
              <dt>Priority</dt><dd>${esc(x.priority)}</dd>
              <dt>Assigned team</dt><dd>${esc(x.assigned_team)}</dd>
              <dt>Created</dt><dd>${esc((x.created_at||'').slice(0,16).replace('T',' '))}</dd>
              <dt>Description</dt><dd>${esc(x.description||'')}</dd>
              ${x.resolution?`<dt>Resolution</dt><dd>${esc(x.resolution)}</dd>`:''}</dl>
          <button style="margin-top:10px" onclick="askAbout('What is the status of ${esc(x.ticket_id)}?')">Ask about this ticket</button>
        </div>`;
        $('#ticketDetail').scrollIntoView({behavior:'smooth', block:'nearest'});
      });
    }
    else if(v==='incidents'){
      const d = await (await fetch('/api/v1/incidents', {headers:H()})).json();
      const t = d.incidents||[];
      el.innerHTML = `<h3>Active incidents</h3><p class="hint">Live from the incident service via the incident MCP server.</p>`
        + (t.length ? `<table><tr><th>ID</th><th>Service</th><th>Sev</th><th>Status</th><th>Affected</th><th>Workaround</th></tr>`
          + t.map((x,i)=>`<tr class="clickable" data-i="${i}"><td>${esc(x.incident_id)}</td><td>${esc(x.service)}</td>
             <td><span class="pill ${esc(x.severity)}">${esc(x.severity)}</span></td>
             <td><span class="pill ${esc(x.status)}">${esc(x.status)}</span></td>
             <td>${esc(x.affected_users)}</td><td>${esc((x.workaround||'').slice(0,90))}</td></tr>`).join('')
          + `</table><div id="incDetail"></div>` : '<div class="empty">No active incidents.</div>');
      el.querySelectorAll('tr.clickable').forEach(tr => tr.onclick = () => {
        const x = t[+tr.dataset.i];
        $('#incDetail').innerHTML = `<div class="detail">
          <h4>${esc(x.incident_id)} — ${esc(x.service)}</h4>
          <dl><dt>Severity</dt><dd>${esc(x.severity)}</dd>
              <dt>Status</dt><dd>${esc(x.status)}</dd>
              <dt>Affected users</dt><dd>${esc(x.affected_users)}</dd>
              <dt>Started</dt><dd>${esc((x.started_at||'').slice(0,16).replace('T',' '))}</dd>
              <dt>Description</dt><dd>${esc(x.description||'')}</dd>
              <dt>Workaround</dt><dd>${esc(x.workaround||'None published')}</dd></dl>
          <button style="margin-top:10px" onclick="askAbout('Am I affected by ${esc(x.incident_id)}?')">Am I affected?</button>
        </div>`;
        $('#incDetail').scrollIntoView({behavior:'smooth', block:'nearest'});
      });
    }
    else if(v==='evaluation'){
      const d = await (await fetch('/api/v1/evaluation')).json();
      if(d.status){ el.innerHTML = `<h3>Evaluation</h3><div class="empty">${esc(d.status)}<br><small>${esc(d.detail||'')}</small></div>`; return; }
      const ex = d.experiments||[];
      el.innerHTML = `<h3>Retrieval evaluation</h3>
        <p class="hint">Measured by an actual run. No placeholder values — if a metric has not been computed it says so.</p>
        <table><tr><th>Experiment</th><th>Recall@10</th><th>Precision@10</th><th>MRR</th><th>nDCG@10</th><th>p95</th></tr>`
        + ex.map(e=>`<tr><td>${esc(e.experiment_id)} ${esc(e.label)}</td>
            <td>${e.overall.recall.toFixed(3)}</td><td>${e.overall.precision.toFixed(3)}</td>
            <td>${e.overall.mrr.toFixed(3)}</td><td><b>${e.overall.ndcg.toFixed(3)}</b></td>
            <td>${e.latency_ms.p95.toFixed(1)} ms</td></tr>`).join('')
        + `</table>
        <h3 style="margin-top:22px">Pairwise significance</h3>
        <p class="hint">Paired bootstrap, 10,000 resamples. An improvement whose CI crosses zero is reported as no measurable difference.</p>
        <table><tr><th>A</th><th>B</th><th>&Delta; nDCG</th><th>95% CI</th><th>Verdict</th></tr>`
        + (d.comparisons||[]).map(c=>`<tr><td>${esc(c.a)}</td><td>${esc(c.b)}</td>
            <td>${c.delta>=0?'+':''}${c.delta.toFixed(4)}</td>
            <td>[${c.ci_low.toFixed(4)}, ${c.ci_high.toFixed(4)}]</td>
            <td>${c.significant?'<b>'+esc(c.verdict)+'</b>':esc(c.verdict)}</td></tr>`).join('')
        + `</table>`;
    }
    else if(v==='system'){
      const m = await (await fetch('/metrics')).json();
      const rd = await (await fetch('/ready')).json();
      const b = m.token_budget||{}, mc = m.mcp||{}, ca = m.cache||{};
      el.innerHTML = `<h3>System status</h3><p class="hint">Live counters from this process.</p>
      <div class="grid">
        <div class="stat"><div class="k">Requests</div><div class="v">${m.requests_total} <small>${m.errors_total} errors</small></div></div>
        <div class="stat"><div class="k">Latency p95</div><div class="v">${m.latency_ms.p95} <small>ms</small></div></div>
        <div class="stat"><div class="k">Tokens today</div><div class="v">${b.tokens_today??'—'} <small>/ ${b.tpd_limit??'—'}</small></div></div>
        <div class="stat"><div class="k">MCP calls</div><div class="v">${mc.calls??0} <small>${mc.fallbacks_to_in_process??0} fallbacks</small></div></div>
        <div class="stat"><div class="k">Cache</div><div class="v">${((ca.hit_rate??0)*100).toFixed(0)}% <small>${ca.llm_calls_avoided??0} calls avoided</small></div></div>
        <div class="stat"><div class="k">Authz denials</div><div class="v">${m.authorization_denials}</div></div>
        <div class="stat"><div class="k">Corpus</div><div class="v">${rd.documents} <small>docs / ${rd.chunks} chunks</small></div></div>
        <div class="stat"><div class="k">Tools</div><div class="v">${rd.tools} <small>${(rd.mcp?.domains||[]).length} MCP domains</small></div></div>
      </div>
      <pre>${esc(JSON.stringify(m,null,2))}</pre>`;
    }
  }catch(e){ el.innerHTML = `<div class="empty">Could not load: ${esc(e.message)}</div>`; }
}

/* ---------- start ---------- */
if(TOKEN) boot(); else { loadAccounts(); }
</script>
</body>
</html>"""
