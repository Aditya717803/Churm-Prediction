import React, { useEffect, useState } from 'react';
import { FiCalendar } from "react-icons/fi";
import { motion } from "framer-motion";
import { BsCloudSunFill } from "react-icons/bs";


function TopBar() {
  const [dateTime, setDateTime] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setDateTime(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formattedDate = dateTime.toLocaleDateString("en-US", {
    weekday: "long",
    year: "numeric",
    month: "short",
    day: "numeric",
  });

  const formattedTime = dateTime.toLocaleTimeString("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="border-b px-4 mb-4 mt-2 pb-4 border-stone-200">
      <div className="flex items-center justify-between p-0.5">
        <div>
          <motion.span 
            className="text-sm font-bold block text-stone-500" 
            initial={{ opacity: 0, y: -10 }} 
            animate={{ opacity: 1, y: 0 }} 
            transition={{ duration: 0.5 }}
          >
            <div className = "flex">
             <div className = "text-2xl text-yellow-300"><BsCloudSunFill /></div><div className = "mt-2"> Good morning,</div>  <span className = "text-2xl text-violet-600"> User! </span>
            </div>
          </motion.span>
          <motion.span 
            className="text-md font-bold block text-stone-500" 
            initial={{ opacity: 0 }} 
            animate={{ opacity: 1 }} 
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {formattedDate} | {formattedTime}
          </motion.span>
        </div>

        <motion.button 
          className="flex text-sm items-center gap-2 bg-stone-100 transition-colors hover:bg-violet-100 hover:text-violet-700 px-3 py-1.5 rounded"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <FiCalendar />
          <span>Prev 6 Months</span>
        </motion.button>
      </div>
    </div>
  );
}

export default TopBar;