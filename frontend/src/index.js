import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);



// import React from "react";
// import { createRoot } from "react-dom/client";
// import App from "./App";

// const root = createRoot(document.getElementById("root"));
// root.render(<App />);


// import {StrictMode} from 'react';
// import {createRoot} from 'react-dom/client';

// import App from './App'

// // This is the ID of the div in your index.html file

// const rootElement = document.getElementById('root');
// const root = createRoot(rootElement);

// // 👇️ if you use TypeScript, add non-null (!) assertion operator
// const root = createRoot(rootElement!);

// root.render(
//   <StrictMode>
//     <App />
//   </StrictMode>,
// );