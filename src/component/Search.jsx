import React from 'react';
import { FiMaximize, FiSearch } from "react-icons/fi";

const Search = () => {
  return (
    <div className="bg-stone-200 mb-4 relative rounded-lg flex items-center px-4 py-2 text-sm">
      <FiSearch className="mr-2 text-stone-600" />
      <input
        type="text"
        placeholder="Search"
        className="w-full bg-transparent placeholder:text-stone-500 focus:outline-none"
      />
      <span className="p-2 text-xs flex gap-0.5 items-center shadow bg-stone-50 rounded absolute right-1.5 top-1/2 -translate-y-1/2">
        <FiMaximize />
      </span>
    </div>
  );
};

export default Search;
