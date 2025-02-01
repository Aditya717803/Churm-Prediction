import React from 'react'
import TopBar from './TopBar'
import Grid from './Grid'
// import Upload from './Upload'
// import Button from './Button'
// import Login from './login'

function Navbar() {
  return (
    <div className='bg-white rounded-lg pb-4 shadow h-[200vh]'>
      <TopBar/>
      <Grid/> 
    </div>
  )
}

export default Navbar
