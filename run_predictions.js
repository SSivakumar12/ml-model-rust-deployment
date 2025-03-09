const fs = require("fs");

const rawData = fs.readFileSync("data/testing_dataset.json", "utf8");
const jsonData = JSON.parse(rawData);

function preprocess_data(data) {
    // Overwrites value of keys to numeric datatype to comply with rust data model
    Object.keys(data).forEach(key => {
        if (typeof data[key] === "boolean") {
            data[key] = data[key] ? 1 : 0;
        }
    });
    return data;
}

async function invoke_model(payload) {
    // predicts the liklihood of surival using particular architecture
    // currently supports decisiontree/logistic classifier.
    try {
        const response = await fetch("http://127.0.0.1:8080/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: Object.values(payload),
                                   model_architecture: "decisiontree"})
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.text();
        return data;
    } catch (error) {
        console.error("Error:", error);
        return null;
    }
}

async function run_predictions() {
    let data = jsonData["x_test"].map(obj => preprocess_data(obj));
    const results = await Promise.all(data.map(invoke_model));
    return results;
}

async function main() {
    // invokes model and predictions and then 
    let startTime = new Date();
    const outputs = await run_predictions();
    let endTime = new Date();
    let timeDiff = (endTime - startTime) / 1000;


    console.log(outputs);
    console.log(`total time taken to make predictions are: ${timeDiff} seconds`)

}

main();
