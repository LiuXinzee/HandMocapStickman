import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import Home from "./pages/Home";
import DataCollect from "./pages/DataCollect";
import Train from "./pages/Train";
import Translate from "./pages/Translate";
import TrainSkeleton from "./pages/TrainSkeleton";
import VirtualMocap from "./pages/VirtualMocap";
import DynamicCollect from "./pages/DynamicCollect";
import DynamicTrain from "./pages/DynamicTrain";
import DynamicTranslate from "./pages/DynamicTranslate";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/collect"} component={DataCollect} />
      <Route path={"/train"} component={Train} />
      <Route path={"/train-skeleton"} component={TrainSkeleton} />
      <Route path={"/translate"} component={Translate} />
      <Route path={"/collect-dynamic"} component={DynamicCollect} />
      <Route path={"/train-dynamic"} component={DynamicTrain} />
      <Route path={"/translate-dynamic"} component={DynamicTranslate} />
      <Route path={"/mocap"} component={VirtualMocap} />
      <Route path={"/404"} component={NotFound} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider defaultTheme="dark">
        <TooltipProvider>
          <Toaster />
          <Router />
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
