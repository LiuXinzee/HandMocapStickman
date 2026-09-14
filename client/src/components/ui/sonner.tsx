import { Toaster as Sonner, type ToasterProps } from "sonner";

/*
 * 原来这里是 `useTheme()` from "next-themes"。**本项目没有挂 next-themes 的
 * provider**（主题走的是 contexts/ThemeContext），所以它永远返回默认值
 * `"system"` —— toast 跟着**操作系统**深浅，在浅色的页面上会弹出一个深色气泡。
 * 全站是固定浅色，这里就写死 light。
 */
const Toaster = ({ ...props }: ToasterProps) => {
  return (
    <Sonner
      theme="light"
      className="toaster group"
      style={
        {
          "--normal-bg": "var(--popover)",
          "--normal-text": "var(--popover-foreground)",
          "--normal-border": "var(--border)",
        } as React.CSSProperties
      }
      {...props}
    />
  );
};

export { Toaster };
