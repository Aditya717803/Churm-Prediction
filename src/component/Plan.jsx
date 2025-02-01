const Plan = () => {
    return (
        <div className="flex sticky top-[calc(100vh_-_48px_-_16px)] flex-col h-12 border-t px-2 border-stone-300 justify-end text-xs">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-bold text-violet-400">Sign out</p>
            <p className="text-stone-500">Back to Home Page</p>
          </div>
  
          <button className="px-2 py-1.5 font-medium bg-violet-200 hover:bg-violet-200 hover:text-black transition-colors rounded">
            Sign Out
          </button>
        </div>
      </div>
    )
  };
  
  export default Plan;
  