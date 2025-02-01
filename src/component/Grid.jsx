import React from 'react'
import StatsCard from './StatsCard'
// import Graph from './Graph1'
import Graph2  from './Graph2'
import Graph3 from './Graph3'
import Graph4 from './Graph4'
import Graph5 from './Graph5'

function Grid() {
  return (
    <div className = "px-4 grid gap-3 grid-cols-12">
      <StatsCard/>
      {/* <Graph/> */}
      <Graph2/>
      <Graph4/>
      <Graph3/>
      <Graph5/>
    </div>
  )
};



export default Grid
