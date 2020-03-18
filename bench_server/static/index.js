function renderChart(data) {
    let chart = new CanvasJS.Chart("chartContainer", {
        animationEnabled: true,
        zoomEnabled: true,
        theme: "dark2",
        title: {
            text: "NNC Benchmarks"
        },
        axisX: {
            valueFormatString: "HH:MM DD MMM"
        },
        axisY: {
            title: "Latency",
            logarithmic: true,
            prefix: "",
            suffix: "us"
        },
        toolTip: {
            //shared: true
        },
        legend: {
            cursor: "pointer",
            itemclick: toogleDataSeries
        },
        data: data
    });
    chart.render();
	function clickHandler(evt) {
		let dp = chart.getDataPointAtXY(evt.x, evt.y);
		let url = "https://github.com/bertmaher/pytorch/commit/" + dp.dataPoint["commit_hash"];
		window.open(url);
	}
	let c_elem = document.getElementById("chartContainer");
	c_elem.onclick = clickHandler;

    function toogleDataSeries(e) {
        if (typeof(e.dataSeries.visible) === "undefined" || e.dataSeries.visible) {
            e.dataSeries.visible = false;
        } else {
            e.dataSeries.visible = true;
        }
        chart.render();
    }

}

function processResponse(str) {
    let ds = JSON.parse(str);
    let benchs = {};
    for (let d of ds) {
		let commit_hash = d[0];
        let date = new Date(d[1] * 1000);
        for (let bench of d[2]) {
            let desc = bench["desc"];
            let us = bench["us"];
            if (!(desc in benchs)) {
                benchs[desc] = [];
            }
            benchs[desc].push({
                x: date,
                y: us,
				commit_hash: commit_hash
            });
        }
    }
    let data = [];
    let first_shown_toggle = true;
    for (let desc in benchs) {
		let dps = benchs[desc];
        data.push({
            type: "line",
            //axisYType: "secondary",
            showInLegend: true,
            visible: first_shown_toggle,
            name: desc,
            markerSize: 0,
            dataPoints: dps
        });
        if (first_shown_toggle) {
            first_shown_toggle = false;
        }
    }
    return data;
}
window.addEventListener('load', function() {
    let xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            let data = processResponse(this.response);
            renderChart(data);
        }
    };
    xhttp.open("GET", "data", true);
    xhttp.send();
});
