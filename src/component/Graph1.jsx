import React from "react";
import ReactECharts from "echarts-for-react";

const Graph1 = () => {
  const options = {
    title: { text: "Sample ML Graph" },
    tooltip: { trigger: "axis" }, // Enables hover effect
    xAxis: { type: "category", data: ["Epoch 1", "Epoch 2", "Epoch 3"] },
    yAxis: { type: "value" },
    series: [
      {
        name: "Accuracy",
        type: "line",
        data: [0.75, 0.82, 0.88],
        smooth: true, // Makes the line smoother
        lineStyle: { width: 3 },
        itemStyle: { color: "#8700ff" }, // Custom color
      },
    ],
  };

  return (
    <div className="col-span-12 overflow-hidden rounded border border-stone-300 p-4">
      <ReactECharts option={options} />
    </div>
  );
};

export default Graph1;
