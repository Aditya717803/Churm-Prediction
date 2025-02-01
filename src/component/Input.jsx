import { useState } from "react";

const FormComponent = () => {
  const [formData, setFormData] = useState({
    gender: "",
    seniorCitizen: "",
    dependents: "",
    phoneService: "",
    onlineSecurity: "",
    onlineBackup: "",
    techsupport: "",
    contract: "",
    paperlessbilling: "",
  });

  // Handle change for all form inputs
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  // Handle form submission and send data to API
  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);

    try {
      const response = await fetch("http://localhost:5000/ml-model", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      console.log("ML Model Response:", data);

    } catch (error) {
      console.error("Error sending data to API:", error);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
      {[
        { label: "Gender", name: "gender", options: ["Male", "Female"] },
        { label: "Senior Citizen", name: "seniorCitizen", options: ["no", "yes"] },
        { label: "Dependents", name: "dependents", options: ["no", "yes"] },
        { label: "Phone Service", name: "phoneService", options: ["no", "yes"] },
        { label: "Online Security", name: "onlineSecurity", options: ["no", "yes"] },
        { label: "Online Backup", name: "onlineBackup", options: ["no", "yes"] },
        { label: "Tech Support", name: "techsupport", options: ["no", "yes"] },
        { label: "Contract", name: "contract", options: ["Month-to-month", "One year", "Two year"] },
        { label: "Paperless Billing", name: "paperlessbilling", options: ["no", "yes"] },
      ].map(({ label, name, options }, index) => (
        <div key={name} className="flex flex-col">
          <label className="text-gray-700 font-medium mb-1">{label}</label>
          <select
            name={name}
            value={formData[name] || ""} // Ensure the value reflects the state (default to empty string)
            onChange={handleChange}
            className="p-2 border rounded-lg focus:ring focus:ring-violet-300"
          >
            <option value="">Select</option> {/* Default empty option */}
            {options.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      ))}

      <div className="col-span-2 flex justify-center mt-6">
        <button
          type="submit"
          className="btn btn-primary max-sm:btn-sm lg:btn-lg shadow-md shadow-purple-200 bg-purple-50 text-purple-700 border border-purple-700 hover:bg-purple-100 hover:text-purple-800"
        >
          Submit
        </button>
      </div>
    </form>
  );
};

export default FormComponent;
