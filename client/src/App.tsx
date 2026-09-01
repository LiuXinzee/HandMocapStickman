import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import NotFound from "@/pages/NotFound";
import { Route, Switch } from "wouter";
import ErrorBoundary from "./components/ErrorBoundary";
import { ThemeProvider } from "./contexts/ThemeContext";
import { GloveProvider } from "./contexts/GloveContext";
import Home from "./pages/Home";
import DataCollect from "./pages/DataCollect";
import Train from "./pages/Train";
import Translate from "./pages/Translate";
import TrainSkeleton from "./pages/TrainSkeleton";
import VirtualMocap from "./pages/VirtualMocap";
import SequenceCollect from "./pages/SequenceCollect";
import CollectSentence from "./pages/CollectSentence";
import TrainSequence from "./pages/TrainSequence";
import TrainSentence from "./pages/TrainSentence";

function Router() {
  return (
    <Switch>
      <Route path={"/"} component={Home} />
      <Route path={"/collect"} component={DataCollect} />
      <Route path={"/collect-seq"} component={SequenceCollect} />
      <Route path={"/collect-sentence"} component={CollectSentence} />
      <Route path={"/train"} component={Train} />
      <Route path={"/train-seq"} component={TrainSequence} />
      <Route path={"/train-sentence"} component={TrainSentence} />
      <Route path={"/train-skeleton"} component={TrainSkeleton} />
      <Route path={"/translate"} component={Translate} />
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
          {/* 手套连接提到 Router 之上：全应用只开一次串口，换页不再重连。
              children 是这里创建的稳定元素，provider 的高频 setState 不会重渲染整棵子树。 */}
          <GloveProvider>
            <Router />
          </GloveProvider>
        </TooltipProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
