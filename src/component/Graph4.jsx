import React from "react";
import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";

function Graph4() {
  const option1 = {
    title: {
      text: "Contract Distribution",
      left: "center",
      textStyle: { color: "#4B0082" },
    },
    tooltip: { trigger: "item" },
    series: [
      {
        type: "pie",
        radius: ["40%", "70%"],
        data: [
          { value: 3000, name: "Month-to-Month", itemStyle: { color: "#8A2BE2" } },
          { value: 2000, name: "One Year", itemStyle: { color: "#9370DB" } },
          { value: 1500, name: "Two Year", itemStyle: { color: "#BA55D3" } },
        ],
        animationDuration: 1000,
      },
    ],
  };
  const option2 = {
    title: {
      text: "Payment Method Distribution",
      left: "center",
      textStyle: { color: "#4B0082" }
    },
    tooltip: { trigger: "item" },
    xAxis: {
      type: "value", // Values will be on the x-axis
    },
    yAxis: {
      type: "category", // Categories will be on the y-axis
      data: ["Credit Card", "Bank Transfer", "E-Wallet", "Cash"], // Payment methods
    },
    series: [
      {
        type: "bar",
        data: [
          { value: 3200, name: "Credit Card", itemStyle: { color: "#8A2BE2" } },
          { value: 2700, name: "Bank Transfer", itemStyle: { color: "#9370DB" } },
          { value: 1800, name: "E-Wallet", itemStyle: { color: "#BA55D3" } },
          { value: 1500, name: "Cash", itemStyle: { color: "#DA70D6" } }
        ],
        animationDuration: 1000,
      }
    ],
  };

  return (
    <div className="flex col-span-12 overflow-hidden rounded border border-stone-300 p-4 py-3 hover:border-purple-600">
      <ReactECharts option={option1} style={{ width: "100%", height: "300px" }} />
      <ReactECharts option={option2} style={{ width: "100%", height: "300px" }} />
    </div>
  );
}

export default Graph4;
