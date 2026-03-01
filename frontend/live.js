(function(){
  const startBtn = document.getElementById('start');
  const stopBtn = document.getElementById('stop');
  const preview = document.getElementById('preview');
  const transcriptDiv = document.getElementById('transcript');
  let stream = null, mr = null, ws = null;

  function appendText(t){
    transcriptDiv.textContent = transcriptDiv.textContent + (transcriptDiv.textContent ? '\n' : '') + t;
    transcriptDiv.scrollTop = transcriptDiv.scrollHeight;
  }

  function stopAll(){
    try{ if (mr && mr.state !== 'inactive') mr.stop(); }catch(e){}
    try{ if (stream) stream.getTracks().forEach(t=>t.stop()); }catch(e){}
    try{
      if (ws && ws.readyState === WebSocket.OPEN){
        try{ ws.send('GET_COMMITTED'); }catch(e){}
        // wait briefly for server to return final transcript
        setTimeout(()=>{ try{ ws.close(); }catch(e){} }, 1200);
      } else {
        try{ if (ws) ws.close(); }catch(e){}
      }
    }catch(e){ try{ if (ws) ws.close(); }catch(e){} }
    preview.srcObject = null; stream = mr = ws = null;
    startBtn.disabled = false; stopBtn.disabled = true;
  }

  function startRecorder(){
    if (!stream) return;
    if (!window.MediaRecorder) { alert('MediaRecorder not supported'); return; }
    const options = { mimeType: 'video/webm' };
    try{ mr = new MediaRecorder(stream, options); } catch(e){ console.error(e); alert('Recorder init failed'); startBtn.disabled = false; return; }
    mr.ondataavailable = (ev)=>{
      if (ev.data && ev.data.size>0 && ws && ws.readyState===WebSocket.OPEN){ ev.data.arrayBuffer().then(buf=>ws.send(buf)); }
    };
    mr.onerror = (e)=>console.error('recorder error', e);
    mr.start(1000);
    stopBtn.disabled = false;
  }

  startBtn.onclick = async ()=>{
    startBtn.disabled = true;
    try{
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      preview.srcObject = stream;
      const proto = (window.location.protocol==='https:') ? 'wss://' : 'ws://';
      ws = new WebSocket(proto + window.location.host + '/ws/video-stream');
      ws.binaryType = 'arraybuffer';
      ws.onmessage = (ev)=>{ try{ const msg = JSON.parse(ev.data); if (msg && msg.text) appendText(msg.text); }catch(e){} };
      ws.onerror = (e)=>console.error('ws error', e);
      ws.onopen = ()=> startRecorder();
      ws.onclose = ()=> stopAll();
    }catch(e){ console.error(e); startBtn.disabled = false; }
  };

  stopBtn.onclick = ()=> stopAll();
})();
