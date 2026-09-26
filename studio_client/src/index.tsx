import ReactDOM from "react-dom/client";
import { Root } from "./App";
import { enableMapSet } from "immer";
import "./StudioChrome.css";

enableMapSet();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <Root />,
);
