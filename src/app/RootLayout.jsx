import React from "react";
import Sidebar from "../component/Sidebar";
import Navbar from "../component/Navbar";
import Input from "../component/Input";

import { Outlet, useLocation } from "react-router-dom";

const RootLayout = () => {
  const location = useLocation(); 

  return (
    <div className="bg-gradient-to-l from-white to-gray-100 min-h-screen">
      <div className="grid grid-cols-[280px,_1fr] gap-6 p-6">
        {/* Sidebar with slight shadow for depth */}
        <Sidebar className="shadow-lg rounded-lg" />
        
        <div className="p-6 bg-white rounded-lg shadow-md">
          {/* Show Navbar only on Dashboard */}
          {location.pathname === "/" && <Navbar />}
          
          <div className="bg-white p-6 rounded-lg shadow-md">
          {/* Render Input.jsx when pathname is /input */}
          {location.pathname === "/input" ? <Input /> : <Outlet />}
        </div>
        </div>
      </div>
    </div>
  );
};

export default RootLayout;
