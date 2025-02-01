import React from "react";
import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";

const GenderBarChart = () => {
  const option1 = {
    title: {
      text: "Gender Distribution",
      left: "center",
      textStyle: { color: "#4B0082", fontSize: 18, fontWeight: "bold" },
    },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: {
      type: "category",
      data: ["Male", "Female"],
      axisLabel: { color: "#4B0082" },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#4B0082" },
    },
    series: [
      {
        data: [3555, 3488],
        type: "bar",
        color: "#8A2BE2",
        animationDuration: 1000,
      },
    ],
  };

  const option2 = {
    title: {
      text: "Senior Citizen Distribution",
      left: "center",
      textStyle: {
        color: "#4B0082",
        fontSize: 18,
        fontWeight: "bold",
      },
      subtext: "Overview of senior vs non-senior population",
      subtextStyle: {
        color: "#666",
        fontSize: 14,
      },
    },
    tooltip: {
      trigger: "item",
      formatter: "{a} <br/>{b}: {c} ({d}%)",
      backgroundColor: "rgba(0, 0, 0, 0.7)",
      borderRadius: 4,
      padding: 10,
      textStyle: {
        color: "#fff",
      },
    },
    series: [
      {
        name: "Senior Citizen Ratio",
        type: "pie",
        radius: ["40%", "70%"],
        data: [
          {
            value: 1200,
            name: "Senior Citizens",
            itemStyle: { color: "#8A2BE2" },
          },
          {
            value: 5843,
            name: "Non-Senior Citizens",
            itemStyle: { color: "#9370DB" },
          },
        ],
        animationDuration: 1000,
        label: {
          show: true,
          position: "outside",
          formatter: "{b}: {c} ({d}%)",
          color: "#333",
          fontSize: 14,
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: "rgba(0, 0, 0, 0.5)",
          },
          label: {
            show: true,
            fontSize: 16,
            fontWeight: "bold",
            color: "#fff",
          },
        },
      },
    ],
  };

  return (
    <div className=" flex col-span-12 overflow-hidden rounded border border-stone-300 p-4 hover:border-purple-500">
    <ReactECharts option={option1} style={{ width: "100%", height: "400px" }} />
    <ReactECharts option={option2} style={{ width: "100%", height: "400px" }} />
  </div>
  );
};

export default GenderBarChart;
