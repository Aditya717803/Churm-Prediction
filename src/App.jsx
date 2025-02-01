import './App.css';
import React from 'react';
import RootLayout from "./app/RootLayout";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./component/Navbar";
import Input from "./component/Input";
// import AdviceSystem from "./pages/AdviceSystem";

function App() {
  return (
    <Router>
      <RootLayout>
        <Routes>
          <Route path="/" element={<Dashboard/>} />
          <Route path="/input" element={<Input/>} />
          {/* <Route path="/advice" element={<AdviceSystem />} /> */}
        </Routes>
      </RootLayout>
    </Router>
  );
}

export default App;
