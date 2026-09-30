// 身份证拼版助手 —— Electron 主进程
// 纯本地离线运行：加载同目录 index.html，AI 识别在渲染进程（wasm）内完成，照片不联网。
const { app, BrowserWindow, Menu, Tray, nativeImage, clipboard, session, dialog, shell } = require('electron');
const path = require('path');

const SELFTEST = process.argv.includes('--selftest');
let win = null;
let tray = null;
let isQuitting = false;
let trayHintShown = false;

// 单实例：重复打开时聚焦已有窗口，而不是再开一个
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) {
      if (!win.isVisible()) win.show();
      if (win.isMinimized()) win.restore();
      win.focus();
    }
  });

  function createWindow() {
    win = new BrowserWindow({
      width: 1440,
      height: 980,
      minWidth: 1080,
      minHeight: 720,
      backgroundColor: '#eef1f5',
      title: '身份证拼版助手',
      autoHideMenuBar: true,
      show: false,
      webPreferences: {
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
        spellcheck: false
      }
    });

    win.removeMenu();
    win.loadFile(path.join(__dirname, 'index.html'));

    win.once('ready-to-show', () => win.show());

    // 点窗口右上角 X：不退出，隐藏到右下角系统托盘后台常驻；托盘菜单“退出”才真正关闭
    win.on('close', (e) => {
      if (!isQuitting) {
        e.preventDefault();
        win.hide();
        if (tray && !trayHintShown) {
          trayHintShown = true;
          try {
            tray.displayBalloon({
              title: '身份证拼版助手仍在后台运行',
              content: '单击右下角托盘图标可重新打开窗口；右键托盘图标选“退出”可关闭程序。'
            });
          } catch (_) {}
        }
      }
    });

    // 外链用系统浏览器打开，不在应用内导航
    win.webContents.setWindowOpenHandler(({ url }) => {
      if (/^https?:/i.test(url)) {
        shell.openExternal(url);
        return { action: 'deny' };
      }
      return { action: 'allow' };
    });

    // 右键菜单：恢复浏览器里的“复制 / 复制图像 / 全选”等（Electron 默认不提供网页右键菜单）
    win.webContents.on('context-menu', (_ev, params) => {
      const ef = params.editFlags || {};
      const tpl = [];
      if (params.mediaType === 'image') {
        tpl.push({
          label: '复制图像',
          click: () => { try { win.webContents.copyImageAt(params.x, params.y); } catch (_) {} }
        });
        tpl.push({ type: 'separator' });
      }
      if (params.isEditable) {
        tpl.push(
          { role: 'cut', label: '剪切', enabled: ef.canCut },
          { role: 'copy', label: '复制', enabled: ef.canCopy },
          { role: 'paste', label: '粘贴', enabled: ef.canPaste },
          { type: 'separator' },
          { role: 'selectAll', label: '全选', enabled: ef.canSelectAll }
        );
      } else {
        tpl.push(
          { role: 'copy', label: '复制', enabled: ef.canCopy },
          {
            label: '复制 A4 拼版图片',
            click: async () => {
              try {
                const d = await win.webContents.executeJavaScript(
                  'document.getElementById("a4-canvas").toDataURL("image/png")'
                );
                clipboard.writeImage(nativeImage.createFromDataURL(d));
              } catch (_) {}
            }
          },
          { type: 'separator' },
          { role: 'selectAll', label: '全选', enabled: ef.canSelectAll }
        );
      }
      Menu.buildFromTemplate(tpl).popup({ window: win });
    });

    if (SELFTEST) {
      const t0 = Date.now();
      win.webContents.on('did-finish-load', () => {
        const iv = setInterval(async () => {
          let r = null;
          try {
            r = await win.webContents.executeJavaScript(
              'JSON.stringify({ready:window.__engineReadyAt||null,title:document.title})'
            );
          } catch (e) { r = null; }
          if (r) {
            try {
              const o = JSON.parse(r);
              if (o.ready) {
                clearInterval(iv);
                console.log('SELFTEST_OK title=' + o.title + ' engineReady=' + o.ready + 'ms total=' + (Date.now() - t0) + 'ms');
                app.exit(0);
              }
            } catch (e) {}
          }
          if (Date.now() - t0 > 90000) {
            clearInterval(iv);
            console.log('SELFTEST_TIMEOUT engine not ready in 90s');
            app.exit(1);
          }
        }, 500);
      });
    }
  }

  function showMainWindow() {
    if (!win) return;
    if (!win.isVisible()) win.show();
    if (win.isMinimized()) win.restore();
    win.focus();
  }

  function createTray() {
    let img = nativeImage.createFromPath(path.join(__dirname, 'assets', 'images', 'tray.png'));
    if (!img || img.isEmpty()) img = nativeImage.createFromPath(path.join(__dirname, 'build', 'icon.png'));
    tray = new Tray(img);
    tray.setToolTip('身份证拼版助手 · 后台运行中（单击恢复窗口）');
    tray.on('click', showMainWindow);
    tray.on('double-click', showMainWindow);
    tray.setContextMenu(Menu.buildFromTemplate([
      { label: '打开 身份证拼版助手', click: showMainWindow },
      { type: 'separator' },
      { label: '退出', click: () => { isQuitting = true; app.quit(); } }
    ]));
  }

  app.whenReady().then(() => {
    // 网页里的“下载 PNG”（a[download]）改为弹出系统“另存为”对话框
    session.defaultSession.on('will-download', (event, item) => {
      const suggested = item.getFilename() || '身份证拼版.png';
      const rc = dialog.showSaveDialogSync(win, {
        title: '保存拼版图片',
        defaultPath: suggested,
        filters: [
          { name: 'PNG 图片', extensions: ['png'] },
          { name: '所有文件', extensions: ['*'] }
        ]
      });
      if (rc) {
        item.setSavePath(rc);
        item.once('done', (_e, state) => {
          if (state === 'completed') console.log('SAVED ' + item.getSavePath());
        });
      } else {
        item.cancel();
      }
    });

    // 精简菜单：打印 / 刷新 / 缩放 / 全屏 / 关于
    const isMac = process.platform === 'darwin';
    const template = [
      ...(isMac ? [{ role: 'appMenu' }] : []),
      {
        label: '文件',
        submenu: [
          { label: '打印…', accelerator: 'Ctrl+P', click: () => win && win.webContents.print() },
          { type: 'separator' },
          isMac ? { role: 'close' } : { role: 'quit', label: '退出' }
        ]
      },
      {
        label: '编辑',
        submenu: [
          { role: 'undo', label: '撤销' }, { role: 'redo', label: '重做' },
          { type: 'separator' },
          { role: 'cut', label: '剪切' }, { role: 'copy', label: '复制' },
          { role: 'paste', label: '粘贴' }, { role: 'selectAll', label: '全选' }
        ]
      },
      {
        label: '视图',
        submenu: [
          { role: 'reload', label: '刷新' },
          { role: 'toggleDevTools', label: '开发者工具' },
          { type: 'separator' },
          { role: 'resetZoom', label: '重置缩放' },
          { role: 'zoomIn', label: '放大' },
          { role: 'zoomOut', label: '缩小' },
          { type: 'separator' },
          { role: 'togglefullscreen', label: '全屏' }
        ]
      },
      {
        label: '帮助',
        submenu: [
          {
            label: '关于 身份证拼版助手',
            click: () => dialog.showMessageBox(win, {
              type: 'info',
              title: '关于',
              message: '身份证拼版助手 2.0',
              detail: '身份证正反面自动拉正 · A4拼版打印\n纯本地离线运行，照片不上传。'
            })
          }
        ]
      }
    ];
    Menu.setApplicationMenu(Menu.buildFromTemplate(template));

    createWindow();
    if (!SELFTEST) createTray();

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
  });

  app.on('before-quit', () => {
    isQuitting = true;
    if (tray) tray.destroy();
  });

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
  });
}
