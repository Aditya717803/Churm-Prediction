import React from "react";
import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";

function Graph5() {
  const option3 = {
    title: {
      text: "Monthly Charges Distribution",
      left: "center",
      textStyle: { color: "#4B0082", fontWeight: "bold", fontSize: 18 },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: {
        type: "shadow",
      },
      backgroundColor: "rgba(0, 0, 0, 0.7)",
      textStyle: {
        color: "#fff",
        fontSize: 14,
      },
    },
    xAxis: {
      type: "value", // Values will be on the x-axis (charges)
      name: "Charges",
      nameTextStyle: {
        color: "#4B0082",
        fontSize: 14,
      },
      axisLabel: {
        formatter: "{value} USD", // Display values with a currency
        color: "#4B0082",
      },
      axisLine: {
        lineStyle: {
          color: "#4B0082",
          width: 2,
        },
      },
    },
    yAxis: {
      type: "category", // Months will be on the y-axis
      data: ["January", "February", "March", "April", "May", "June"], // Months
      axisLabel: {
        color: "#4B0082",
      },
      axisLine: {
        lineStyle: {
          color: "#4B0082",
          width: 2,
        },
      },
    },
    series: [
      {
        type: "line",
        data: [
          { value: 500, name: "January", itemStyle: { color: "#8A2BE2" } },
          { value: 600, name: "February", itemStyle: { color: "#9370DB" } },
          { value: 750, name: "March", itemStyle: { color: "#BA55D3" } },
          { value: 800, name: "April", itemStyle: { color: "#DA70D6" } },
          { value: 950, name: "May", itemStyle: { color: "#FF69B4" } },
          { value: 1100, name: "June", itemStyle: { color: "#D2691E" } },
        ],
        smooth: true, // Smooth line curve
        animationDuration: 1000,
        lineStyle: {
          width: 3,
          type: "solid", // Line style
        },
        areaStyle: {
          color: "rgba(138, 43, 226, 0.3)", // Gradient area fill
        },
        itemStyle: {
          borderWidth: 2,
          borderColor: "#fff", // White border around the data points
        },
      },
    ],
    // Adding a gradient color for the line
    color: ["#8A2BE2", "#9370DB", "#BA55D3", "#DA70D6", "#FF69B4", "#D2691E"],
  };

  return (
    <div className="flex col-span-12 overflow-hidden rounded border border-stone-300 p-4 shadow-md hover:border hover:border-violet-600">
      <ReactECharts option={option3} style={{ width: "100%", height: "340px" }} />
    </div>
  );
}

export default Graph5;
