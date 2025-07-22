'use client';
// page.tsx
import React, { useEffect, useState } from "react";
import dynamic from 'next/dynamic';
import axios from "axios";

// Dynamically import ReactApexChart with no SSR since it requires window
const ReactApexChart = dynamic(() => import('react-apexcharts'), { ssr: false });

function CandlestickChart() {
  const [seriesData, setSeriesData] = useState([]);

  useEffect(() => {
    // Fetch data client-side
    const fetchData = async () => {
      try {
        const res = await axios.get("http://localhost:8000/api/chart", { params: { symbol: "NEPSE" }});
        const { o, h, l, c, t } = res.data;

        const formattedData = t.map((timestamp: number, i: number) => ({
          x: new Date(timestamp * 1000),
          y: [o[i], h[i], l[i], c[i]]
        }));

        setSeriesData(formattedData);
      } catch (err) {
        console.error("Error fetching data:", err);
        setSeriesData([]);
      }
    };

    fetchData();
  }, []);

  const chartOptions: ApexCharts.ApexOptions = {
    chart: {
      type: 'candlestick' as const,
      height: 350,
      background: "#fff"
    },
    title: {
      text: "Dynamic Candlestick Chart",
      align: "left"
    },
    xaxis: {
      type: "datetime"
    },
    yaxis: {
      tooltip: {
        enabled: true
      }
    }
  };

  return (
    <div id="chart">
      <ReactApexChart
        options={chartOptions}
        series={[{ data: seriesData }]}
        type="candlestick"
        height={350}
      />
    </div>
  );
}

export default CandlestickChart;
