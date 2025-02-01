import React from "react";
import * as echarts from "echarts";
import ReactECharts from "echarts-for-react";

function Graph3() {

    const option1 = {
        title: {
          text: "Internet Service Distribution",
          left: "center",
          textStyle: { color: "#4B0082" }
        },
        tooltip: { trigger: "item" },
        series: [{
          type: "funnel",
          left: "10%",
          top: "10%",
          bottom: "10%",
          width: "80%",
          min: 0,
          max: 100,
          minSize: "10%",
          maxSize: "100%",
          sort: "descending",
          gap: 2,
          label: {
            show: true,
            position: "inside",
            color: "#ffffff"
          },
          labelLine: {
            length: 10,
            lineStyle: {
              width: 1,
              type: "solid"
            }
          },
          itemStyle: {
            borderColor: "#fff",
            borderWidth: 1
          },
          data: [
            { value: 70, name: "Fiber Optic", itemStyle: { color: "#8A2BE2" } },
            { value: 50, name: "DSL", itemStyle: { color: "#9370DB" } },
            { value: 30, name: "Cable", itemStyle: { color: "#BA55D3" } },
            { value: 20, name: "Satellite", itemStyle: { color: "#DA70D6" } }
          ],
          animationDuration: 1000,
        }]
      };


  return (
      <div className=" flex col-span-12 overflow-hidden rounded border border-stone-300 p-4 hover:border-purple-600">
      <ReactECharts option={option1} style={{ width: "100%", height: "350px" }} />
    </div>
    );
}
  export default Graph3;