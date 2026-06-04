# Runbook: Gestión de permisos, roles y acceso a recursos

**Categoría:** acceso  
**Aplica a:** DataVault Pro, CloudSync Enterprise, SecureLogin Hub

---

## Usuario sin acceso a recursos después de cambio de departamento

Cuando un usuario cambia de departamento, los permisos no se transfieren automáticamente.

**Pasos:**
1. El nuevo responsable del usuario debe solicitar el acceso a través del portal de RRHH o directamente en el panel de administración.
2. Administración → Usuarios → Seleccionar usuario → Grupos y permisos.
3. Añadir al grupo del nuevo departamento y eliminar del grupo anterior si ya no corresponde.
4. Los cambios tardan hasta 5 minutos en propagarse. Pedir al usuario que cierre sesión y vuelva a entrar.
5. Verificar acceso a carpetas compartidas del nuevo equipo: Recursos compartidos → Permisos → Añadir usuario.

---

## Error "No autorizado" al intentar exportar o ver reportes

**Diagnóstico:**
1. Verificar el rol actual del usuario: debe tener al menos rol "Analyst" para exportar.
2. Comprobar si hay restricciones a nivel de proyecto o carpeta que anulen los permisos del rol.
3. Revisar si la organización tiene habilitada la política "Exportación solo desde IP corporativa" — el usuario puede estar fuera de la oficina.

**Resolución:**
- Para cambiar el rol: Administración → Usuarios → Rol → Seleccionar "Analyst" o superior.
- Para acceso excepcional fuera de IP corporativa: el administrador puede añadir una excepción temporal (máx. 7 días) en Seguridad → Políticas de red.

---

## Cuenta compartida: contraseña cambiada sin aviso

Situación crítica que puede afectar a todo un equipo:
1. Como administrador, resetear la contraseña inmediatamente desde el panel.
2. Notificar a todos los usuarios de la cuenta compartida.
3. Revisar el log de accesos para identificar quién hizo el cambio y desde qué IP.
4. Recomendar migrar a cuentas individuales con permisos compartidos — las cuentas compartidas son un riesgo de seguridad.
5. Si el cambio fue malintencionado, escalar a seguridad con el log de accesos.

---

## Reseteo masivo de acceso (post-migración)

Para restablecer acceso a múltiples usuarios tras una migración:
1. Exportar la lista de usuarios afectados en CSV desde el panel antiguo.
2. Usar el endpoint `POST /admin/users/bulk-reset` con el CSV.
3. Los usuarios recibirán email de activación. El enlace expira en 48h.
4. Para migraciones de más de 200 usuarios, coordinar con el equipo de infraestructura para evitar saturar el servidor de correo.

---

## Escalado

Escalar a Nivel 2 si:
- Se detecta acceso no autorizado (cambio de contraseña desde IP desconocida).
- Un administrador no puede gestionar permisos (problema de privilegios en el backend).
- La política de permisos del tenant está corrupta o devuelve errores inesperados.
