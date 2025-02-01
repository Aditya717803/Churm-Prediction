import { BsCheckCircleFill } from "react-icons/bs";

const ResponsiveButton = () => {
    return (
      <div className="flex justify-center items-center mt-6">
        <button className="btn btn-primary max-sm:btn-sm lg:btn-lg shadow-md shadow-purple-200 bg-purple-50 text-purple-700 border border-purple-700 hover:bg-purple-100 hover:text-purple-800">
        <BsCheckCircleFill />
         Submit
        </button>
      </div>
    );
  };
  
  export default ResponsiveButton;
  