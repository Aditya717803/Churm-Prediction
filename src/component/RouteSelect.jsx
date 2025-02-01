import React from "react";
import { FiBarChart, FiCheckSquare, FiUsers } from "react-icons/fi";
import { Link, useLocation } from "react-router-dom";

export const RouteSelect = () => {
  const location = useLocation();

  return (
    <div className="space-y-1">
      <Route Icon={FiBarChart} path="/" selected={location.pathname === "/"} title="Dashboard" />
      <Route Icon={FiUsers} path="/input" selected={location.pathname === "/input"} title="Predicted Churn" />
      <Route Icon={FiCheckSquare} path="/advice" selected={location.pathname === "/advice"} title="Advice System" />
    </div>
  );
};

const Route = ({ selected, Icon, title, path }) => {
  return (
    <Link to={path}>
      <button
        className={`flex items-center justify-start gap-2 w-full rounded px-2 py-3 text-sm transition-[box-shadow,_background-color,_color] ${
          selected
            ? "bg-white text-stone-950 shadow"
            : "hover:bg-stone-200 bg-transparent text-stone-500 shadow-none"
        }`}
      >
        <Icon className={selected ? "text-violet-800" : ""} />
        <span>{title}</span>
      </button>
    </Link>
  );
};
