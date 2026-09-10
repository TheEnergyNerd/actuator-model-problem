"""Render-only Isaac/Usd styling. No collision or actuator changes."""
import re


def configure_camera(cfg):
    cfg.viewer.eye = (0.43, 0.26, 0.78)
    cfg.viewer.lookat = (0.0, -0.12, 0.51)
    cfg.viewer.resolution = (1920, 1080)
    cfg.scene.dome_light.spawn.color = (0.035, 0.045, 0.065)


def apply_materials():
    import omni.usd
    from pxr import Sdf, Usd, UsdShade
    stage = omni.usd.get_context().get_stage()

    def material(name, color, metallic, roughness):
        path = '/World/AtlasLooks/' + name
        mat = UsdShade.Material.Define(stage, path)
        shader = UsdShade.Shader.Define(stage, path + '/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        return mat

    body = material('MachinedAluminium', (0.62, 0.67, 0.60), 0.35, 0.42)
    pads = material('GripPads', (0.025, 0.035, 0.045), 0.05, 0.75)
    robot = stage.GetPrimAtPath('/World/envs/env_0/Robot')
    if not robot.IsValid():
        raise RuntimeError('Studio robot prim missing')
    bound = []
    for prim in Usd.PrimRange(robot):
        name = prim.GetName()
        match = re.search(r'link_(\d+)', name)
        if match or name in ('base_link', 'palm_link'):
            mat = pads if match and int(match.group(1)) in (3, 7, 11, 15) else body
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            bound.append(str(prim.GetPath()))
    print(f'STUDIO_MATERIALS {len(bound)} links', flush=True)
    return bound
