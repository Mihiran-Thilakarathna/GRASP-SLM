package lk.ac.rjt.grasp;

import java.util.Arrays;

import lk.ac.rjt.grasp.cli.ExportCommand;
import org.springframework.boot.WebApplicationType;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.builder.SpringApplicationBuilder;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.context.ConfigurableApplicationContext;

/**
 * Entry point. Runs the web application by default; {@code java -jar grasp-slm.jar export ...}
 * runs the ingestion/retrieval exporter used to build the training data, so that training and
 * inference see exactly the same chunks and prompts.
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class GraspApplication {

    public static void main(String[] args) {
        if (args.length > 0 && ExportCommand.NAME.equals(args[0])) {
            String[] rest = Arrays.copyOfRange(args, 1, args.length);
            try (ConfigurableApplicationContext ctx = new SpringApplicationBuilder(GraspApplication.class)
                    .web(WebApplicationType.NONE)
                    .run(rest)) {
                int code = ctx.getBean(ExportCommand.class).run(rest);
                System.exit(code);
            }
            return;
        }
        new SpringApplicationBuilder(GraspApplication.class).run(args);
    }
}
