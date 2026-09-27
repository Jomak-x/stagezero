import { ViewerContext } from "./ViewerContext";
import { ThemeConfigurationMessage } from "./WebsocketMessages";
import {
  Burger,
  Button,
  Container,
  Group,
  Paper,
  Box,
  Switch,
  Tooltip,
  useMantineColorScheme,
  useMantineTheme,
  Portal,
} from "@mantine/core";
import {
  IconBrandGithub,
  IconFileDescription,
  IconKeyboard,
  IconMoon,
  IconSun,
} from "@tabler/icons-react";
import { useDisclosure } from "@mantine/hooks";
import { useContext, useId } from "react";

// Type helpers.
type ArrayElement<ArrayType extends readonly unknown[]> =
  ArrayType extends readonly (infer ElementType)[] ? ElementType : never;
type TitlebarContent = NonNullable<
  ThemeConfigurationMessage["titlebar_content"]
>;
function assertUnreachable(x: never): never {
  throw new Error("Didn't expect to get here", x);
}

function getIcon(
  icon: ArrayElement<NonNullable<TitlebarContent["buttons"]>>["icon"],
) {
  let Icon = null;
  switch (icon) {
    case null:
      break;
    case "GitHub":
      Icon = IconBrandGithub;
      break;
    case "Description":
      Icon = IconFileDescription;
      break;
    case "Keyboard":
      Icon = IconKeyboard;
      break;
    default:
      assertUnreachable(icon);
  }
  return Icon;
}

// We inherit props directly from message contents.
export function TitlebarButton(
  props: ArrayElement<NonNullable<TitlebarContent["buttons"]>>,
) {
  const Icon = getIcon(props.icon);
  return (
    <Button
      component="a"
      variant="default"
      size="compact-sm"
      href={props.href || undefined}
      target="_blank"
      leftSection={Icon === null ? null : <Icon size="1em" />}
      ml="xs"
      color="gray"
    >
      {props.text}
    </Button>
  );
}

export function MobileTitlebarButton(
  props: ArrayElement<NonNullable<TitlebarContent["buttons"]>>,
) {
  const Icon = getIcon(props.icon);
  return (
    <Button
      m="sm"
      component="a"
      variant="default"
      href={props.href || undefined}
      target="_blank"
      leftSection={Icon === null ? null : <Icon size="1.5em" />}
      ml="sm"
      color="gray"
    >
      {props.text}
    </Button>
  );
}

export function TitlebarImage(
  props: NonNullable<TitlebarContent["image"]>,
  colorScheme: string,
) {
  let imageSource: string;
  if (props.image_url_dark == null || colorScheme === "light") {
    imageSource = props.image_url_light;
  } else {
    imageSource = props.image_url_dark;
  }
  const image = (
    <img
      src={imageSource}
      alt={props.image_alt}
      style={{
        height: "1.8em",
        margin: "0 0.5em",
      }}
    />
  );

  if (props.href == null) {
    return image;
  }
  return (
    <a style={{ display: "block", height: "1.8em" }} href={props.href}>
      {image}
    </a>
  );
}

export function Titlebar() {
  const viewer = useContext(ViewerContext)!;
  const content = viewer.useGui((state) => state.theme.titlebar_content);
  const darkMode = viewer.useGui((state) => state.theme.dark_mode);
  const titlebarDarkModeCheckboxUuid = viewer.useGui(
    (state) => state.theme.titlebar_dark_mode_checkbox_uuid,
  );
  const colorScheme = useMantineColorScheme().colorScheme;
  const theme = useMantineTheme();

  const [burgerOpen, burgerHandlers] = useDisclosure(false);
  const mobileNavigationId = useId();

  const onDarkModeChange = (checked: boolean) => {
    viewer.useGui.setState({
      theme: {
        ...viewer.useGui.getState().theme,
        dark_mode: checked,
      },
    });
    if (
      titlebarDarkModeCheckboxUuid &&
      viewer.mutable.current?.sendMessage != null
    ) {
      viewer.mutable.current.sendMessage({
        type: "GuiUpdateMessage",
        uuid: titlebarDarkModeCheckboxUuid,
        updates: { value: checked },
      });
    }
  };

  if (content == null) {
    return null;
  }

  const buttons = content.buttons;
  const imageData = content.image;
  const showDarkModeToggle = Boolean(titlebarDarkModeCheckboxUuid);

  return (
    <Box
      className="sz-titlebar"
      data-theme={darkMode ? "dark" : "light"}
      style={{
        margin: 0,
        border: "0",
        zIndex: 10,
      }}
    >
      <Paper className="sz-titlebar-surface" shadow="0 0 0.8em 0 rgba(0,0,0,0.1)" style={{ height: "100%" }}>
        <Container
          fluid
          className="sz-titlebar-content"
          style={() => ({
            height: "100%",
            display: "flex",
            alignItems: "center",
          })}
        >
          <Group className="sz-titlebar-brand" style={() => ({ marginRight: "auto" })}>
            {imageData !== null ? TitlebarImage(imageData, colorScheme) : null}
            {content.title_text != null && content.title_text !== "" ? (
              <Box
                component="span"
                className="sz-titlebar-name"
                style={{
                  fontFamily: theme.fontFamily,
                  color: colorScheme === "dark" ? "#ffffff" : undefined,
                }}
              >
                {content.title_text}
              </Box>
            ) : null}
          </Group>
          <Group
            className="sz-titlebar-actions"
            display={{ base: "none", sm: "flex" }}
            style={() => ({
              flexWrap: "nowrap",
            })}
          >
            {buttons?.map((btn, index) => (
              <TitlebarButton {...btn} key={index} />
            ))}
            {showDarkModeToggle ? (
              <Tooltip
                label={darkMode ? "Light mode" : "Dark mode"}
                withinPortal
              >
                <Switch
                  size="sm"
                  ml="xs"
                  checked={darkMode}
                  onChange={(e) => onDarkModeChange(e.currentTarget.checked)}
                  thumbIcon={
                    darkMode ? (
                      <IconMoon size="0.75rem" />
                    ) : (
                      <IconSun size="0.75rem" />
                    )
                  }
                  color="gray"
                  aria-label={darkMode ? "Switch to light mode" : "Switch to dark mode"}
                />
              </Tooltip>
            ) : null}
          </Group>
          <Burger
            size="sm"
            opened={burgerOpen}
            onClick={burgerHandlers.toggle}
            title={!burgerOpen ? "Open navigation" : "Close navigation"}
            aria-label={!burgerOpen ? "Open navigation" : "Close navigation"}
            aria-expanded={burgerOpen}
            aria-controls={mobileNavigationId}
            display={{ base: "block", sm: "none" }}
          ></Burger>
        </Container>
        <Portal>
          {burgerOpen && <Paper
            id={mobileNavigationId}
            className="sz-titlebar-menu"
            data-theme={darkMode ? "dark" : "light"}
            component="nav"
            aria-label="Studio navigation"
            display={{ base: "flex", sm: "none" }}
            radius="0"
          >
            {buttons?.map((btn, index) => (
              <MobileTitlebarButton {...btn} key={index} />
            ))}
            {showDarkModeToggle ? (
              <Box m="sm" style={{ display: "flex", alignItems: "center" }}>
                <Switch
                  size="sm"
                  label={darkMode ? "Dark mode" : "Light mode"}
                  checked={darkMode}
                  onChange={(e) => onDarkModeChange(e.currentTarget.checked)}
                  thumbIcon={
                    darkMode ? (
                      <IconMoon size="0.75rem" />
                    ) : (
                      <IconSun size="0.75rem" />
                    )
                  }
                  color="gray"
                />
              </Box>
            ) : null}
          </Paper>}
        </Portal>
      </Paper>
    </Box>
  );
}
