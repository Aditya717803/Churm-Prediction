import React from "react";
import { motion } from "framer-motion";
import { Accounttogggle } from "./accounttogggle.jsx";
import Search from "./Search.jsx";
import { RouteSelect } from "/src/component/RouteSelect.jsx";
import Plan from "/src/component/Plan.jsx";

function Sidebar() {
  return (
    <motion.div
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <div className=" sticky top-4 h-[calc(100vh-32px-48px)]">
        <Accounttogggle />
        <Search />
        <RouteSelect />
      </div>
      <Plan />
    </motion.div>
  );
}

export default Sidebar;
