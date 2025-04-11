export function modifyConfig(config: Config): Config {
  // Assuming config.models is an array of objects with 'title' and 'model' properties
  config.models.forEach(model => {
    model.title = "test"; // Set a default title if not already set
    model.model = "test"; // Set the model to 'test'
  });

  console.log(config); // Print the modified config to the console
  return config;
}

