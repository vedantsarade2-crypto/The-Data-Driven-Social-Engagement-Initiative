function formatNumber(value){
    return Number(value || 0).toLocaleString("en-IN", {maximumFractionDigits: 0});
}

function renderBars(targetId, items, formatter){
    const el = document.getElementById(targetId);
    if(!el) return;
    if(!items || !items.length){
        el.innerHTML = '<div class="empty">No chart data available.</div>';
        return;
    }
    const max = Math.max(...items.map(x => Number(x.value) || 0), 1);
    el.innerHTML = items.map(item => {
        const width = Math.max(3, ((Number(item.value) || 0) / max) * 100);
        return `
            <div class="bar-row">
                <div class="bar-label" title="${escapeHtml(item.label)}">${escapeHtml(item.label)}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div>
                <div class="bar-value">${formatter ? formatter(item.value) : item.value}</div>
            </div>`;
    }).join("");
}

function renderOverviewCharts(data){
    renderBars("topicChart", data.topic || [], v => formatNumber(v));

    const el = document.getElementById("viralChart");
    if(!el) return;
    const items = data.viral || [];
    const total = items.reduce((a,b) => a + Number(b.value || 0), 0);
    if(!total){
        el.innerHTML = '<div class="empty">No prediction data available.</div>';
        return;
    }
    const colors = ["#4f6df5","#8b5cf6","#f39a43"];
    let start = 0;
    const segments = [];
    items.forEach((item,i)=>{
        const angle = Number(item.value)/total*360;
        segments.push(`${colors[i%colors.length]} ${start}deg ${start+angle}deg`);
        start += angle;
    });
    el.innerHTML = `
        <div class="donut" style="background:conic-gradient(${segments.join(",")})">
            <div class="donut-center">${total}<small>videos</small></div>
        </div>
        <div class="legend">
            ${items.map((item,i)=>`
                <div class="legend-item">
                    <span class="legend-dot" style="background:${colors[i%colors.length]}"></span>
                    ${escapeHtml(item.label)} <strong>${item.value}</strong>
                </div>`).join("")}
        </div>`;
}

function renderAudienceCharts(data){
    renderBars("sentimentChart", data.sentiment || [], v => Number(v).toFixed(3));
}

function renderTrendChart(data){
    const el = document.getElementById("trendChart");
    if(!el) return;
    const items = data.trend || [];
    if(!items.length){
        el.innerHTML = '<div class="empty">No trend data available.</div>';
        return;
    }
    const max = Math.max(...items.map(x=>Number(x.value)||0), 0.001);
    el.innerHTML = items.map(item=>{
        const h = Math.max(5, (Number(item.value)/max)*175);
        return `
            <div class="line-col">
                <div class="line-value">${Number(item.value).toFixed(2)}</div>
                <div class="line-bar" style="height:${h}px"></div>
                <div class="line-label">${escapeHtml(item.label)}</div>
            </div>`;
    }).join("");
}

function initContentSearch(){
    const input = document.getElementById("contentSearch");
    const table = document.getElementById("contentTable");
    const count = document.getElementById("contentCount");
    if(!input || !table) return;

    const rows = Array.from(table.querySelectorAll("tbody tr"));
    function update(){
        const q = input.value.trim().toLowerCase();
        let visible = 0;
        rows.forEach(row=>{
            const show = row.innerText.toLowerCase().includes(q);
            row.style.display = show ? "" : "none";
            if(show) visible++;
        });
        if(count) count.textContent = `${visible} records`;
    }
    input.addEventListener("input", update);
}

function escapeHtml(value){
    return String(value ?? "")
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;")
        .replaceAll('"',"&quot;")
        .replaceAll("'","&#039;");
}
