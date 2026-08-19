import { WorkbenchProvider } from './context';
import { WorkbenchShell } from './components/layout';

export default function App() {
  return (
    <WorkbenchProvider>
      <WorkbenchShell />
    </WorkbenchProvider>
  );
}
