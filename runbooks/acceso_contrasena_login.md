# Runbook: Problemas de contraseña y login bloqueado

**Categoría:** acceso  
**Aplica a:** SecureLogin Hub, DataVault Pro, CloudSync Enterprise

---

## Diagnóstico inicial

Antes de actuar, recopilá la siguiente información:
- ¿El usuario nunca pudo entrar, o funcionaba antes?
- ¿El error es "contraseña incorrecta", "cuenta bloqueada" o "usuario no encontrado"?
- ¿Cuántos intentos fallidos hubo? (revisar logs de autenticación)
- ¿Es una cuenta individual o compartida?

---

## Cuenta bloqueada por intentos fallidos

El sistema bloquea automáticamente tras 5 intentos fallidos consecutivos.

**Pasos de resolución:**
1. Acceder al panel de administración → Gestión de usuarios → Buscar por email.
2. Verificar el estado: si aparece "Bloqueado (intentos)", usar el botón "Desbloquear cuenta".
3. Notificar al usuario por email con instrucciones para cambiar la contraseña.
4. Si el bloqueo se repite en menos de 24h, escalar a seguridad — podría ser un intento de acceso no autorizado.

---

## Restablecimiento de contraseña

**Para el usuario:**
1. Ir a la pantalla de login → "¿Olvidaste tu contraseña?".
2. Ingresar el email corporativo. El enlace de reseteo llega en menos de 2 minutos.
3. El enlace es válido por 30 minutos. Si expira, repetir el proceso.
4. La nueva contraseña debe cumplir: mínimo 10 caracteres, una mayúscula, un número, un símbolo.

**Para el administrador (reseteo masivo):**
- Usar la API: `POST /admin/users/bulk-reset` con lista de IDs.
- Máximo 50 usuarios por lote. Para más de 50, usar el script de migración disponible en la wiki.
- Todos los usuarios recibirán un email con enlace de activación válido por 48h.

---

## Contraseña correcta pero no entra

Si el usuario jura que la contraseña es correcta:
1. Verificar que Bloq Mayús no esté activado.
2. Revisar si hay una sesión activa en otro dispositivo que bloqueó la cuenta.
3. Comprobar si el email de la cuenta tiene alias o variantes (ej: con/sin punto).
4. Revisar si el tenant tiene autenticación SSO activa — en ese caso el login directo está deshabilitado y deben usar el portal corporativo.

---

## Escalado

Escalar a Nivel 2 si:
- El usuario sigue sin acceso después de reseteo.
- El error es "Account suspended" (requiere intervención del equipo de compliance).
- Más de 3 usuarios del mismo tenant reportan el problema simultáneamente.
