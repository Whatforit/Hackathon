import { createRequire } from "module";
const require = createRequire(import.meta.url);
const countries = require("./fun-data.json")
const percentiles = require("./fun-data-percentile.json")
// import percentiles from './fun-data-percentile.json';

//Country Schema:
/*
  {
    'countryname': {
      "property_one": "value",
      "property_two": "value",
      ...
    }
  }
*/
//Percentile Schema:
/*
  {
    "property_one": {"90th Percentile": "value"},
    "property_one": {"90th Percentile": "value"},
    ...
  }
*/

json_generator(countries, percentiles)

//create json for webpage
// function json_generator(countries, percentiles){
//   let countries_array = Object.entries(countries)
//   for(let i=0; i<countries_array.length; i++){
//     const name = countries_array[0];
//     const properties = countries_array[1];
//     const realPrediction = predict(properties)
//     for (const property in properties) {
//       if (!percentiles[property]) continue;
//       //90th percentile val
//       imputeVal = percentiles[property]["90th Percentile"]
      
    
//     }
//   }
// }
function predict(){
  //gets prediction for val
}